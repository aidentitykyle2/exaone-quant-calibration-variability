#!/usr/bin/env python3
"""
final_analysis.py — 마무리 분석 세 가지

A. 지표별 꼬리 집중도
   Choice-KL이 한국어에서는 가장 민감하고 영어에서는 가장 둔한 역전이 관찰되었다.
   가설: KL은 유계가 아니라 소수 문항이 평균을 지배하므로, 효과가 작을 때 꼬리 잡음에
   묻힌다. 상위 10% 문항이 전체 합계에서 차지하는 비중으로 확인한다.
   (판정 불일치율은 문항마다 0~1로 유계라 꼬리 영향이 제한된다.)

B. 필요 시드 수 (2단계 부트스트랩 기준)
   기존 추정치는 시드 변동만 고려한 근사였다. 여기서는 논문에서 실제로 쓰는 절차
   그대로, 문항과 시드를 함께 재표집하면서 시드 수 k를 바꿔가며 95% 구간이 0을
   배제하는 최소 k를 찾는다.
   한계: 보유 시드(11개)에서 복원추출로 더 많은 시드를 흉내 내므로, 큰 k의 추정은
   실제보다 낙관적일 수 있다. 표에 보유 시드 수를 함께 밝힌다.

C. 동률률과 저하율의 상관
   한국어 정확도가 유독 불안정한 이유를 찾는다. 시드마다 동률률이 크게 흔들리므로
   (KMMLU 15~25%), 동률률이 저하율과 연동되는지 확인한다.

사용법:
  python final_analysis.py --base quant_experiments/EXAONE-3.5-7.8B-Instruct
"""

import argparse
import csv
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from m4_full import load_ll  # noqa: E402
from paired_bootstrap import load_item_scores  # noqa: E402

BENCH = {"KMMLU": ("ko", "kmmlu_"), "MMLU": ("en", "mmlu_")}
CONDS = ("ko", "en_instruct")
DIR_RE = re.compile(r"^gptq_(ko|en_instruct)_s(\d+)$")   # n=256만 (크기 실험 제외)


def metrics_matrix(ref_ll, q_ll, items):
    """문항별 Choice-KL과 판정 불일치율을 한 번에 계산(벡터화)."""
    R = np.array([ref_ll[i] for i in items])
    Q = np.array([q_ll[i] for i in items])
    lb = R - np.logaddexp.reduce(R, axis=1, keepdims=True)
    lq = Q - np.logaddexp.reduce(Q, axis=1, keepdims=True)
    kl = (np.exp(lb) * (lb - lq)).sum(1)
    mb = R == R.max(1, keepdims=True)
    mq = Q == Q.max(1, keepdims=True)
    flip = 1.0 - (mb & mq).sum(1) / (mb.sum(1) * mq.sum(1))
    return kl, flip, mb.sum(1) > 1, mq.sum(1) > 1


def min_seeds(colmeans_a, colmeans_b, expect_positive, rng, iters, kmax=40):
    """
    2단계 재표집으로 시드 수 k별 95% 구간을 구하고, 0을 배제하는 최소 k를 찾는다.
    colmeans_*: (iters, 시드수) — 반복마다 문항을 재표집해 구한 시드별 평균.
    """
    sa, sb = colmeans_a.shape[1], colmeans_b.shape[1]
    out = {}
    for k in range(2, kmax + 1):
        ca = rng.integers(0, sa, (iters, k))
        cb = rng.integers(0, sb, (iters, k))
        stat = (np.take_along_axis(colmeans_a, ca, 1).mean(1)
                - np.take_along_axis(colmeans_b, cb, 1).mean(1))
        lo, hi = np.quantile(stat, [0.025, 0.975])
        out[k] = (float(np.median(stat)), float(lo), float(hi))
        if (lo > 0) if expect_positive else (hi < 0):
            return k, out[k]
    return None, out.get(kmax)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--iters", type=int, default=1000)
    ap.add_argument("--top-frac", type=float, default=0.10)
    args = ap.parse_args()
    base = Path(args.base)
    rng = np.random.default_rng(7)
    out_dir = base / "final"
    out_dir.mkdir(parents=True, exist_ok=True)

    lines, rows = [], []
    store = {}     # (bench, cond) -> dict(seeds, kl, flip, tie, acc_D)

    for bench, (lang, prefix) in BENCH.items():
        root = base / "seed_eval" / lang
        ref_ll = load_ll(root / "bf16", prefix)
        bf_acc = load_item_scores(root / "bf16", prefix, "tie_split")
        runs = {}
        for d in sorted(root.iterdir()):
            m = DIR_RE.match(d.name) if d.is_dir() else None
            if m:
                runs.setdefault(m.group(1), {})[int(m.group(2))] = d
        if not runs:
            lines.append(f"[{bench}] gptq 시드 디렉터리를 찾지 못했다.")
            continue

        items = sorted(set(ref_ll) & set(bf_acc))
        for c in CONDS:
            seeds = sorted(runs.get(c, {}))
            kl_cols, fl_cols, ties, accs = [], [], [], []
            for s in seeds:
                q_ll = load_ll(runs[c][s], prefix)
                q_acc = load_item_scores(runs[c][s], prefix, "tie_split")
                common = [i for i in items if i in q_ll and i in q_acc]
                kl, fl, _, tq = metrics_matrix(ref_ll, q_ll, common)
                kl_cols.append(kl)
                fl_cols.append(fl)
                ties.append(float(tq.mean()))
                accs.append(np.array([q_acc[i] for i in common]))
            store[(bench, c)] = {
                "seeds": seeds, "items": common,
                "kl": np.column_stack(kl_cols), "flip": np.column_stack(fl_cols),
                "tie": np.array(ties), "acc": np.column_stack(accs),
                "bf": np.array([bf_acc[i] for i in common]),
            }
            lines.append(f"[{bench}] gptq[{c}] 시드 {seeds} · 문항 {len(common)}")
    lines.append("")

    # ---------------- A. 꼬리 집중도 ----------------
    lines.append(f"=== A. 꼬리 집중도: 상위 {args.top_frac:.0%} 문항이 합계에서 차지하는 비중 ===")
    lines.append("    비중이 높을수록 소수 문항이 평균을 지배해 시드 간 변동이 커진다.")
    lines.append(f"  {'벤치마크':9s} {'조건':12s} {'Choice-KL':>10s} {'판정 불일치':>12s}")
    for (bench, c), d in store.items():
        share = {}
        for key in ("kl", "flip"):
            v = np.sort(d[key], axis=0)[::-1]
            k = max(1, int(args.top_frac * v.shape[0]))
            share[key] = float((v[:k].sum(0) / v.sum(0)).mean())
        lines.append(f"  {bench:9s} {c:12s} {share['kl'] * 100:9.1f}% {share['flip'] * 100:11.1f}%")
        rows.append(["tail_share", bench, c, "", round(share["kl"], 4), round(share["flip"], 4)])
    lines.append("")

    # ---------------- B. 필요 시드 수 ----------------
    lines.append("=== B. 필요 시드 수 (2단계 부트스트랩 95% 구간이 0을 배제하는 최소 시드) ===")
    lines.append("    Δ_KO = (en_instruct 캘리브 − ko 캘리브), KMMLU에서 측정 → 양수 예상")
    lines.append("    Δ_EN = (ko 캘리브 − en_instruct 캘리브), MMLU에서 측정 → 양수 예상")
    lines.append("    Accuracy는 BF16 대비 저하율의 차이(%p), 나머지는 BF16 대비 지표의 차이")
    lines.append(f"  {'평가':7s} {'지표':16s} {'현재 추정':>10s} {'필요 시드':>9s} {'보유':>5s}")

    for bench, (worse, better) in (("KMMLU", ("en_instruct", "ko")), ("MMLU", ("ko", "en_instruct"))):
        A, B = store.get((bench, worse)), store.get((bench, better))
        if not A or not B:
            continue
        n = len(A["items"])
        W = np.array([rng.multinomial(n, np.full(n, 1.0 / n)) for _ in range(args.iters)]) / n
        for label, key in (("Choice-KL", "kl"), ("Decision Fidelity", "flip"), ("Accuracy", "acc")):
            if key == "acc":
                bf = W @ A["bf"]
                da = ((bf[:, None] - (W @ A["acc"])) / bf[:, None]) * 100
                db = ((bf[:, None] - (W @ B["acc"])) / bf[:, None]) * 100
            else:
                da, db = W @ A[key], W @ B[key]
            k, (med, lo, hi) = min_seeds(da, db, True, rng, args.iters)
            lines.append(f"  {bench:7s} {label:16s} {med:+10.4f} {str(k) if k else '>40':>9s} "
                         f"{len(A['seeds']):5d}")
            rows.append(["min_seeds", bench, label, k or "", round(med, 5),
                         f"[{lo:.5f},{hi:.5f}]"])
    lines.append("    주의: 보유 시드에서 복원추출로 더 많은 시드를 흉내 내므로, 보유 수를 크게")
    lines.append("          넘어서는 값은 낙관적일 수 있다.")
    lines.append("")

    # ---------------- C. 동률률과 저하율 ----------------
    lines.append("=== C. 동률률과 BF16 대비 저하율의 상관 (시드 간) ===")
    lines.append("    한국어 정확도가 유독 불안정한 이유를 동률 구조 변화로 설명할 수 있는지 본다.")
    lines.append(f"  {'벤치마크':9s} {'조건':12s} {'동률률 범위':>16s} {'저하율 범위':>14s} {'상관':>7s}")
    allpts = []
    for (bench, c), d in store.items():
        acc = d["acc"].mean(0)
        deg = (d["bf"].mean() - acc) / d["bf"].mean() * 100
        tie = d["tie"]
        r = float(np.corrcoef(tie, deg)[0, 1]) if len(tie) > 2 else float("nan")
        allpts += list(zip(tie, deg))
        lines.append(f"  {bench:9s} {c:12s} {tie.min() * 100:6.1f}~{tie.max() * 100:5.1f}% "
                     f"{deg.min():6.2f}~{deg.max():5.2f}% {r:7.2f}")
        rows.append(["tie_corr", bench, c, len(tie), round(r, 4), ""])
    if len(allpts) > 2:
        t, g = np.array(allpts).T
        lines.append(f"  {'전체 통합':9s} {'':12s} {'':16s} {'':14s} "
                     f"{np.corrcoef(t, g)[0, 1]:7.2f}  (점 {len(allpts)}개)")
    lines.append("    상관이 높으면 시드 변동이 '모델이 얼마나 확신하는가'의 변화와 연동된다는 뜻이다.")

    text = "\n".join(lines)
    (out_dir / "final_summary.txt").write_text(text + "\n", encoding="utf-8")
    with open(out_dir / "final_results.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["analysis", "bench", "key", "n", "value", "extra"])
        w.writerows(rows)
    print(text)
    print(f"\n저장 → {out_dir}")


if __name__ == "__main__":
    main()
