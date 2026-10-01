#!/usr/bin/env python3
"""
paired_bootstrap.py — 캘리브레이션 언어 효과의 신뢰구간 추정

lm-evaluation-harness의 --log_samples 출력(문항별 정오)을 읽어 두 가지 불확실성을
분리해 보고한다.

  (a) 문항 부트스트랩      : 평가 문항 표집 변동. 시드를 고정하고 문항만 재표집한다.
  (b) 2단계 부트스트랩     : 문항 표집 + 캘리브레이션 표본(시드) 변동을 함께 반영한다.

두 가지를 함께 보고하는 이유
  (a)만 보고하면 캘리브레이션 표본 추출 변동이 빠진 좁은 구간이 나와 과신을
  유도한다. Williams & Aletras(ACL 2024)는 같은 원천에서 뽑은 캘리브레이션 세트끼리도
  다운스트림 성능이 상당히 변동함을 보고했다. (b)가 이 변동까지 포함한 구간이다.

추정 대상 (m = 양자화 기법)
  D(m, c, B)  = (acc_BF16 − acc_m,c) / acc_BF16 × 100      [BF16 대비 저하율, %]
  L_KO(m)     = D(m, en_instruct, KMMLU) − D(m, ko, KMMLU)   [한국어 평가 언어 효과, 양수 예상]
  L_EN(m)     = D(m, en_instruct, MMLU)  − D(m, ko, MMLU)    [영어 평가 언어 효과, 음수 예상]
  X(m)        = L_KO − L_EN                                  [교차 크기, 양수 예상]

  en_instruct를 대조군으로 쓰는 이유: 도메인(instruction)을 고정한 순수 언어 비교다.

대응(paired) 구조
  같은 벤치마크 안에서는 모든 모델이 같은 문항 재표집을 공유한다(문항 난이도 통제).
  KMMLU와 MMLU는 서로 다른 문항 집합이므로 독립적으로 재표집한다.
  ko와 en_instruct의 시드는 서로 독립적인 추출이므로 조건별로 따로 재표집한다.

사용법:
  python paired_bootstrap.py --base quant_experiments/EXAONE-3.5-7.8B-Instruct
"""

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

SAMPLE_RE = re.compile(r"samples_(.+?)_(\d{4}-\d{2}-\d{2}T[\d\-.]+)\.jsonl$")
BENCH = {"KMMLU": ("ko", "kmmlu_"), "MMLU": ("en", "mmlu_")}
METHODS = ("gptq", "awq")
CALIBS = ("ko", "en_instruct")


def _continuations(r):
    """기록의 arguments에서 선택지별 이어쓰기 문자열을 순서대로 꺼낸다.
    lm-eval 버전에 따라 리스트 또는 딕셔너리 형식이다."""
    args = r.get("arguments")
    if isinstance(args, dict):
        keys = sorted(args, key=lambda k: int(re.sub(r"\D", "", k) or 0))
        seq = [args[k] for k in keys]
    elif isinstance(args, list):
        seq = args
    else:
        return None
    out = []
    for e in seq:
        if isinstance(e, dict):
            out.append(e.get("arg_1"))
        elif isinstance(e, (list, tuple)) and len(e) >= 2:
            out.append(e[1])
        else:
            return None
    return out


def target_index(r):
    """정답을 선택지 번호로 바꾼다. 정수면 그대로, 문자열(예: '(A)')이면 선택지와 대조한다."""
    t = r.get("target")
    if isinstance(t, int):
        return t
    try:
        return int(t)
    except (TypeError, ValueError):
        pass
    conts = _continuations(r)
    if conts:
        ts = str(t).strip()
        for i, cont in enumerate(conts):
            if cont is not None and str(cont).strip() == ts:
                return i
    return None


def item_score(r, scoring):
    """
    문항 하나의 점수.
      tie_split : 최대 로그우도 동률 선택지에 점수를 나눠 준다(정답이 동률 k개 중 하나면 1/k).
                  선택지 순서에 치우치지 않는 기준이다.
      first     : 동률이면 첫 번째 선택지를 고른다. lm-eval의 argmax 채점과 같다.
      lm_eval   : 기록에 저장된 acc(또는 acc_norm)를 그대로 쓴다.

    vLLM은 로짓을 bf16으로 계산해 선택지 로그우도가 자주 동률이 된다
    (7.8B 실측: KMMLU 19~27%, MMLU 11~14%). first 규칙은 이를 A 쪽으로 해소하므로
    정확도가 선택지 순서에 따라 왜곡되고, 양자화가 동률률을 바꾸므로 조건 간 비교도
    오염된다. 본 연구의 주 기준은 tie_split이다.
    """
    if scoring == "lm_eval":
        v = r.get("acc_norm", r.get("acc"))
        return None if v is None else float(v)
    fr = r.get("filtered_resps")
    if not fr:
        return None
    ll = [float(x[0]) if isinstance(x, (list, tuple)) else float(x) for x in fr]
    t = target_index(r)
    if t is None:
        return None          # 정답을 해석할 수 없는 문항은 제외
    mx = max(ll)
    if scoring == "first":
        return float(ll.index(mx) == t)
    tied = [i for i, v in enumerate(ll) if v == mx]
    return (1.0 if t in tied else 0.0) / len(tied)


def load_item_scores(model_dir: Path, prefix: str, scoring: str = "tie_split") -> dict:
    """(task, doc_id) -> 점수. 같은 과제의 파일이 여럿이면 가장 최근 것을 쓴다."""
    latest = {}
    for p in model_dir.rglob("samples_*.jsonl"):
        m = SAMPLE_RE.search(p.name)
        if not m or not m.group(1).startswith(prefix):
            continue
        task, ts = m.group(1), m.group(2)
        if task not in latest or ts > latest[task][0]:
            latest[task] = (ts, p)
    scores = {}
    for task, (_, p) in latest.items():
        with open(p, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                val = item_score(r, scoring)
                if val is None:
                    continue
                scores[(task, r["doc_id"])] = val
    return scores


def discover(base: Path, lang: str):
    """seed_eval/<lang>/ 아래 모델 디렉터리를 찾는다. bf16과 <method>_<calib>_s<seed>."""
    root = base / "seed_eval" / lang
    found = {}
    if not root.is_dir():
        return found
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        if d.name == "bf16":
            found["bf16"] = d
            continue
        m = re.match(r"^(gptq|awq)_(ko|en_instruct)_s(\d+)$", d.name)
        if m:
            found[(m.group(1), m.group(2), int(m.group(3)))] = d
    return found


def build_matrix(base: Path, bench: str, scoring: str = "tie_split"):
    lang, prefix = BENCH[bench]
    dirs = discover(base, lang)
    if "bf16" not in dirs:
        return None, None, None
    raw = {k: load_item_scores(d, prefix, scoring) for k, d in dirs.items()}
    raw = {k: v for k, v in raw.items() if v}
    common = set.intersection(*(set(v) for v in raw.values()))
    if not common:
        return None, None, None
    items = sorted(common)
    keys = list(raw)
    mat = np.array([[raw[k][it] for k in keys] for it in items], dtype=np.float64)
    dropped = {k: len(v) - len(items) for k, v in raw.items() if len(v) != len(items)}
    return mat, keys, dropped


def degradation(acc_bf16, acc):
    return (acc_bf16 - acc) / acc_bf16 * 100.0


def effects(accs, keys, seed_pick):
    """accs: 모델별 정확도 벡터. seed_pick: {(m,c): [시드 목록]} → 조건별 D 평균."""
    idx = {k: i for i, k in enumerate(keys)}
    b = accs[idx["bf16"]]
    out = {}
    for m in METHODS:
        d = {}
        for c in CALIBS:
            seeds = seed_pick.get((m, c), [])
            vals = [degradation(b, accs[idx[(m, c, s)]]) for s in seeds if (m, c, s) in idx]
            if not vals:
                break
            d[c] = float(np.mean(vals))
        if len(d) == 2:
            out[m] = d["en_instruct"] - d["ko"]
    return out


def ci(samples, alpha=0.05):
    a = np.asarray(samples)
    return float(np.quantile(a, alpha / 2)), float(np.quantile(a, 1 - alpha / 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=12345, help="부트스트랩 난수 시드")
    ap.add_argument("--scoring", choices=["tie_split", "first", "lm_eval"], default="tie_split",
                    help="문항 채점 방식. 기본 tie_split(동률 분할). first/lm_eval은 lm-eval 보고치와 대조용")
    args = ap.parse_args()
    base = Path(args.base)
    out_dir = base / "bootstrap"
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    data = {}
    for bench in BENCH:
        mat, keys, dropped = build_matrix(base, bench, args.scoring)
        if mat is None:
            print(f"[{bench}] 문항별 결과가 없거나 BF16이 없다. --log_samples 평가 완료 여부를 확인할 것.")
            sys.exit(1)
        data[bench] = (mat, keys)
        print(f"[{bench}] 공통 문항 {mat.shape[0]}개, 모델 {len(keys)}개")
        if dropped:
            print(f"  경고: 문항 수가 다른 모델이 있어 공통 문항만 사용 → {dropped}")

    seeds_avail = defaultdict(set)
    for bench, (_, keys) in data.items():
        for k in keys:
            if k != "bf16":
                seeds_avail[(k[0], k[1])].add(k[2])
    seeds_avail = {k: sorted(v) for k, v in seeds_avail.items()}
    print("가용 시드:", {f"{m}/{c}": s for (m, c), s in seeds_avail.items()})

    # 점추정 (전체 문항)
    point_acc = {b: data[b][0].mean(0) for b in data}

    lines = []
    rows = []

    # ---- 정합성 점검: 문항 평균 정확도 (기존 보고치와 대조용) ----
    lines.append(f"=== 채점 방식: {args.scoring} ===")
    lines.append("=== 정합성 점검: seed 0 재평가의 문항 평균 정확도 ===")
    if args.scoring == "tie_split":
        lines.append("    동률 분할 기준값(KMMLU: BF16 0.4669, GPTQ[ko] 0.4525, AWQ[ko] 0.4281)과 일치해야 한다")
    else:
        lines.append("    lm-eval 보고치(KMMLU: BF16 0.4555, GPTQ[ko] 0.4484, AWQ[ko] 0.4200)와 일치해야 한다")
    for b in data:
        keys = data[b][1]
        for k in keys:
            if k == "bf16" or k[2] == 0:
                name = "bf16" if k == "bf16" else f"{k[0]}[{k[1]}]"
                lines.append(f"  {b:5s} {name:22s} {point_acc[b][keys.index(k)]:.4f}")
    lines.append("")

    # ---- 시드별 점추정 ----
    lines.append("=== 시드별 언어 효과 점추정 (%p) ===")
    per_seed = defaultdict(list)
    all_seeds = sorted({s for v in seeds_avail.values() for s in v})
    for s in all_seeds:
        pick = {k: [s] for k in seeds_avail}
        e_ko = effects(point_acc["KMMLU"], data["KMMLU"][1], pick)
        e_en = effects(point_acc["MMLU"], data["MMLU"][1], pick)
        for m in METHODS:
            if m in e_ko and m in e_en:
                per_seed[m].append((s, e_ko[m], e_en[m], e_ko[m] - e_en[m]))
                lines.append(f"  seed {s} {m.upper():4s}  L_KO={e_ko[m]:+.2f}  "
                             f"L_EN={e_en[m]:+.2f}  X={e_ko[m] - e_en[m]:+.2f}")
    lines.append("")
    lines.append("=== 시드 간 평균 ± 표준편차 (%p) ===")
    for m, vals in per_seed.items():
        a = np.array([v[1:] for v in vals])
        sd = a.std(0, ddof=1) if len(a) > 1 else np.full(3, np.nan)
        lines.append(f"  {m.upper():4s} n={len(a)}  L_KO={a[:, 0].mean():+.2f}±{sd[0]:.2f}  "
                     f"L_EN={a[:, 1].mean():+.2f}±{sd[1]:.2f}  X={a[:, 2].mean():+.2f}±{sd[2]:.2f}")
    lines.append("")

    # ---- 부트스트랩 ----
    def run(two_stage: bool, fixed_seed=None):
        acc_b = {m: {"L_KO": [], "L_EN": [], "X": []} for m in METHODS}
        for _ in range(args.iters):
            accs = {}
            for b, (mat, _) in data.items():
                n = mat.shape[0]
                w = rng.multinomial(n, np.full(n, 1.0 / n))   # 문항 재표집(모델 간 공유)
                accs[b] = (w @ mat) / n
            if two_stage:
                pick = {k: list(rng.choice(v, size=len(v), replace=True)) for k, v in seeds_avail.items()}
            else:
                pick = {k: [fixed_seed] for k in seeds_avail}
            e_ko = effects(accs["KMMLU"], data["KMMLU"][1], pick)
            e_en = effects(accs["MMLU"], data["MMLU"][1], pick)
            for m in METHODS:
                if m in e_ko and m in e_en:
                    acc_b[m]["L_KO"].append(e_ko[m])
                    acc_b[m]["L_EN"].append(e_en[m])
                    acc_b[m]["X"].append(e_ko[m] - e_en[m])
        return acc_b

    expect = {"L_KO": ">0", "L_EN": "<0", "X": ">0"}

    def report(title, res, tag):
        lines.append(f"=== {title} (반복 {args.iters}회, 95% 백분위 구간) ===")
        for m in METHODS:
            if not res[m]["X"]:
                continue
            for q in ("L_KO", "L_EN", "X"):
                v = np.array(res[m][q])
                lo, hi = ci(v)
                wrong = float((v <= 0).mean()) if expect[q] == ">0" else float((v >= 0).mean())
                verdict = "구간이 0을 포함하지 않음" if (lo > 0 or hi < 0) else "구간이 0을 포함"
                lines.append(f"  {m.upper():4s} {q:4s} 중앙값={np.median(v):+.2f}  "
                             f"[{lo:+.2f}, {hi:+.2f}]  예상 반대방향 확률={wrong:.3f}  ({verdict})")
                rows.append([tag, m, q, round(float(np.median(v)), 4),
                             round(lo, 4), round(hi, 4), round(wrong, 4)])
        lines.append("")

    if 0 in all_seeds:
        report("(a) 문항 부트스트랩 — seed 0 고정", run(False, fixed_seed=0), "item_only_seed0")
    if len(all_seeds) > 1:
        report("(b) 2단계 부트스트랩 — 문항 + 시드", run(True), "two_stage")
    else:
        lines.append("(b) 2단계 부트스트랩은 시드가 2개 이상일 때 계산한다.")

    # 시드 변동의 크기 판정: 시드 간 표준편차를 문항 부트스트랩 표준오차와 비교한다.
    # (a)와 (b)의 구간 폭을 직접 비교하면 안 된다. (a)는 단일 시드의 추정치이고
    # (b)는 여러 시드의 평균이라 추정 대상이 다르다. 평균이 분산을 줄이므로 (b)가
    # 더 좁게 나올 수도 있다.
    if 0 in all_seeds and len(all_seeds) > 1:
        lines.append("=== 캘리브레이션 표본 변동의 크기 ===")
        lines.append("    시드 간 SD가 문항 부트스트랩 SE와 비슷하거나 크면, 캘리브레이션 표본 추출이")
        lines.append("    평가 문항 표집만큼 결과를 흔든다는 뜻이다(Williams & Aletras 2024의 우려가 현실화).")
        res_a = run(False, fixed_seed=0)
        for m in METHODS:
            if m not in per_seed or len(per_seed[m]) < 2 or not res_a[m]["X"]:
                continue
            a = np.array([v[1:] for v in per_seed[m]])
            for j, q in enumerate(("L_KO", "L_EN", "X")):
                sd_seed = float(a[:, j].std(ddof=1))
                se_item = float(np.std(res_a[m][q], ddof=1))
                ratio = sd_seed / se_item if se_item > 0 else float("nan")
                lines.append(f"  {m.upper():4s} {q:4s} 시드간 SD={sd_seed:.2f}  문항 SE={se_item:.2f}  "
                             f"비율={ratio:.2f}")
        lines.append("")

    lines.append("해석 지침")
    lines.append("  (a) 단일 시드(seed 0) 결과의 불확실성 — 평가 문항 표집만 반영")
    lines.append("  (b) 시드 평균 결과의 불확실성 — 문항 표집 + 캘리브레이션 표본 추출 반영")
    lines.append("  두 구간은 추정 대상이 달라 폭을 직접 비교하지 않는다. 시드 변동의 크기는 위의 비율로 판단한다.")
    lines.append("  논문에는 (b)를 주 결과로, (a)를 단일 시드 대조로 보고한다.")

    text = "\n".join(lines)
    (out_dir / f"bootstrap_summary_{args.scoring}.txt").write_text(text + "\n", encoding="utf-8")
    with open(out_dir / f"bootstrap_ci_{args.scoring}.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["mode", "method", "quantity", "median", "ci_low", "ci_high", "p_opposite"])
        w.writerows(rows)
    print(text)
    print(f"\n저장 → {out_dir}")


if __name__ == "__main__":
    main()
