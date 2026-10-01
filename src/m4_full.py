#!/usr/bin/env python3
"""
m4_full.py — 전체 규모 출력 충실도 분석 (M4, 동률 고려)

128문항 0-shot으로 수행한 m4_kl.py는 검정력이 부족했다(언어당 약 784문항 필요).
lm-eval의 --log_samples 기록에는 각 문항의 선택지별 로그우도가 남아 있으므로,
추가 GPU 연산 없이 실제 평가와 같은 5-shot 조건에서 전체 문항을 분석할 수 있다.
  KMMLU 35,030문항 / MMLU 14,042문항 / seed 0·1·2

측정량 (각 문항에서 BF16과 양자화 모델 비교)
  kl_choice  네 선택지 로그우도를 재정규화한 분포의 KL(P_BF16 || P_quant)
  flip       판정 불일치 확률. 동률을 무작위로 해소한다고 가정할 때
             두 모델이 다른 선택지를 고를 확률 = 1 − |S_B ∩ S_Q| / (|S_B|·|S_Q|)
             (S = 최대 로그우도 동률 집합)

  flip을 동률 확률로 정의한 이유: vLLM의 bf16 로짓으로 동률이 KMMLU 19~27%에서
  발생한다. 첫 번째 선택 규칙으로 판정하면 동률 해소 방식이 뒤집힘 여부를 좌우해
  측정이 오염된다.

교차 판정 (양수가 예상 방향)
  Δ_KO = 지표(en_instruct 캘리브, KMMLU) − 지표(ko 캘리브, KMMLU)
  Δ_EN = 지표(ko 캘리브, MMLU)           − 지표(en_instruct 캘리브, MMLU)

불확실성
  시드 평균 지표에 대해 문항 단위 대응 부트스트랩 구간을 구하고,
  시드별 점추정도 함께 보고한다.

사용법:
  python m4_full.py --base quant_experiments/EXAONE-3.5-7.8B-Instruct
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np

SAMPLE_RE = re.compile(r"samples_(.+?)_(\d{4}-\d{2}-\d{2}T[\d\-.]+)\.jsonl$")
BENCH = {"KMMLU": ("ko", "kmmlu_"), "MMLU": ("en", "mmlu_")}
METHODS = ("gptq", "awq")
CALIBS = ("ko", "en_instruct")


def load_ll(model_dir: Path, prefix: str) -> dict:
    """(task, doc_id) -> 선택지 로그우도 배열. 과제별로 가장 최근 파일을 쓴다."""
    latest = {}
    for p in model_dir.rglob("samples_*.jsonl"):
        m = SAMPLE_RE.search(p.name)
        if not m or not m.group(1).startswith(prefix):
            continue
        if m.group(1) not in latest or m.group(2) > latest[m.group(1)][0]:
            latest[m.group(1)] = (m.group(2), p)
    out = {}
    for task, (_, p) in latest.items():
        with open(p, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                fr = r.get("filtered_resps")
                if not fr:
                    continue
                out[(task, r["doc_id"])] = np.array(
                    [float(x[0]) if isinstance(x, (list, tuple)) else float(x) for x in fr])
    return out


def per_item(ref: np.ndarray, q: np.ndarray):
    """(kl_choice, flip, 기준 동률 여부, 비교 동률 여부)"""
    lb = ref - np.logaddexp.reduce(ref)
    lq = q - np.logaddexp.reduce(q)
    kl = float((np.exp(lb) * (lb - lq)).sum())
    sb = set(np.flatnonzero(ref == ref.max()).tolist())
    sq = set(np.flatnonzero(q == q.max()).tolist())
    flip = 1.0 - len(sb & sq) / (len(sb) * len(sq))
    return kl, flip, len(sb) > 1, len(sq) > 1


def boot_ci(x, iters, rng):
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    means = np.empty(iters)
    for i in range(iters):
        means[i] = x[rng.integers(0, n, n)].mean()
    return float(x.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def boot_two_stage(A, B, iters, rng):
    """문항과 시드를 함께 재표집한다. A, B: (문항, 시드) 행렬. 문항 재표집은 공유(대응),
    시드는 조건별로 독립 재표집(ko와 en_instruct의 시드는 서로 독립 추출이므로)."""
    n, sa, sb = A.shape[0], A.shape[1], B.shape[1]
    out = np.empty(iters)
    for i in range(iters):
        it = rng.integers(0, n, n)
        ra = A[:, rng.integers(0, sa, sa)].mean(1)
        rb = B[:, rng.integers(0, sb, sb)].mean(1)
        out[i] = (ra[it] - rb[it]).mean()
    return float(A.mean() - B.mean()), float(np.quantile(out, 0.025)), float(np.quantile(out, 0.975))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--iters", type=int, default=2000)
    args = ap.parse_args()
    base = Path(args.base)
    rng = np.random.default_rng(12345)

    lines = ["=== M4 전체 규모: 5-shot 선택지 로그우도 기반 출력 충실도 (동률 고려) ===", ""]
    tie_lines = ["--- 동률률 (최대 로그우도가 2개 이상의 선택지에서 같은 문항 비율) ---"]
    metrics = {}   # (bench, method, calib) -> {'kl': item×seed 배열, 'flip': ...}
    seeds_used = {}

    for bench, (lang, prefix) in BENCH.items():
        root = base / "seed_eval" / lang
        ref = load_ll(root / "bf16", prefix)
        if not ref:
            print(f"[{bench}] BF16 기록이 없다. 먼저 BF16 재평가가 필요하다.")
            return
        tie_lines.append(f"  {bench} BF16: {np.mean([len(v) > 1 and (v == v.max()).sum() > 1 for v in ref.values()]) * 100:.2f}%")

        for meth in METHODS:
            for cal in CALIBS:
                runs = []
                for d in sorted(root.glob(f"{meth}_{cal}_s*")):
                    m = re.match(rf"^{meth}_{cal}_s(\d+)$", d.name)
                    if not m:
                        continue
                    q = load_ll(d, prefix)
                    if q:
                        runs.append((int(m.group(1)), q))
                if not runs:
                    continue
                items = sorted(set(ref).intersection(*[set(q) for _, q in runs]))
                kl = np.zeros((len(items), len(runs)))
                fl = np.zeros_like(kl)
                for j, (s, q) in enumerate(runs):
                    ties_q = 0
                    for i, it in enumerate(items):
                        kl[i, j], fl[i, j], _, tq = per_item(ref[it], q[it])
                        ties_q += tq
                    tie_lines.append(f"  {bench} {meth}[{cal}] seed{s}: {ties_q / len(items) * 100:.2f}%")
                metrics[(bench, meth, cal)] = {"items": items, "kl": kl, "flip": fl}
                seeds_used[(bench, meth, cal)] = [s for s, _ in runs]

    lines += tie_lines + [""]
    lines.append("--- 조건별 평균 (시드 평균) ---")
    for (bench, meth, cal), v in sorted(metrics.items()):
        lines.append(f"  {bench:5s} {meth.upper():4s}[{cal:11s}] 시드{seeds_used[(bench, meth, cal)]}  "
                     f"kl_choice={v['kl'].mean():.4f}  flip={v['flip'].mean():.4f}  (문항 {len(v['items'])})")
    lines.append("")

    lines.append("--- 교차 판정: 95% 구간 (문항만 = 시드 평균에 대한 문항 부트스트랩, "
                 "2단계 = 문항 + 시드) ---")
    lines.append("    논문의 주 결과는 2단계 구간이다. 문항만 구간은 시드 변동을 빠뜨려 과신을 유도한다.")
    lines.append("    Δ_KO = 지표(en_instruct 캘리브, KMMLU) − 지표(ko 캘리브, KMMLU)  → 양수 예상")
    lines.append("    Δ_EN = 지표(ko 캘리브, MMLU) − 지표(en_instruct 캘리브, MMLU)    → 양수 예상")
    for meth in METHODS:
        for metric, name in (("kl", "kl_choice"), ("flip", "flip")):
            parts = []
            for bench, worse, better, lab in (("KMMLU", "en_instruct", "ko", "Δ_KO"),
                                              ("MMLU", "ko", "en_instruct", "Δ_EN")):
                a, b = metrics.get((bench, meth, worse)), metrics.get((bench, meth, better))
                if a is None or b is None:
                    parts.append(f"{lab}=계산 불가")
                    continue
                common = sorted(set(a["items"]) & set(b["items"]))
                ia = {it: k for k, it in enumerate(a["items"])}
                ib = {it: k for k, it in enumerate(b["items"])}
                A = a[metric][[ia[it] for it in common]]
                Bm = b[metric][[ib[it] for it in common]]
                m, lo, hi = boot_ci(A.mean(1) - Bm.mean(1), args.iters, rng)
                _, lo2, hi2 = boot_two_stage(A, Bm, args.iters, rng)
                tag = lambda l, h: "0 미포함" if (l > 0 or h < 0) else "0 포함"
                parts.append(f"{lab}={m:+.4f} 문항만[{lo:+.4f},{hi:+.4f}]{tag(lo, hi)} "
                             f"2단계[{lo2:+.4f},{hi2:+.4f}]{tag(lo2, hi2)}")
            lines.append(f"  {meth.upper():4s} {name:9s} " + "   ".join(parts))
    lines.append("")

    lines.append("--- 시드별 점추정 (시드 간 일관성 확인) ---")
    for meth in METHODS:
        for metric, name in (("kl", "kl_choice"), ("flip", "flip")):
            for bench, worse, better, lab in (("KMMLU", "en_instruct", "ko", "Δ_KO"),
                                              ("MMLU", "ko", "en_instruct", "Δ_EN")):
                a, b = metrics.get((bench, meth, worse)), metrics.get((bench, meth, better))
                if a is None or b is None:
                    continue
                sa, sb = seeds_used[(bench, meth, worse)], seeds_used[(bench, meth, better)]
                vals = [f"s{s}={a[metric][:, sa.index(s)].mean() - b[metric][:, sb.index(s)].mean():+.4f}"
                        for s in sorted(set(sa) & set(sb))]
                lines.append(f"  {meth.upper():4s} {name:9s} {lab}: " + "  ".join(vals))
    lines.append("")
    lines.append("주의: 시드별 비교는 같은 번호의 시드끼리 짝지었으나, ko와 en_instruct의 시드는")
    lines.append("      서로 독립적인 추출이므로 짝 자체에 의미는 없다. 부호가 일관되는지만 본다.")

    out = base / "mechanism" / "m4_full_summary.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(lines)
    out.write_text(text + "\n", encoding="utf-8")
    print(text)
    print(f"\n저장 → {out}")


if __name__ == "__main__":
    main()
