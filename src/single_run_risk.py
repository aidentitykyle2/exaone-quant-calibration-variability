#!/usr/bin/env python3
"""
single_run_risk.py — 단일 캘리브레이션 실행의 위험도 정량화 (E1, RQ2 겸용)

단일 실행 연구는 캘리브레이션 세트를 한 번 뽑아 양자화하고, 평가 문항 부트스트랩으로
유의성을 판정한다. 이 스크립트는 이미 확보한 다중 시드 결과로 그 절차를 모든 가능한
시드 짝에 대해 재현해, 단일 실행 분석이 얼마나 자주 잘못된 결론에 도달하는지 센다.

두 종류의 짝
  교차 짝 (ko seed i, en_instruct seed j)
      실제 연구가 하는 비교. 부호가 짝마다 뒤집히는 비율을 본다.
  귀무 짝 (ko seed i, ko seed j) 및 (en_instruct seed i, en_instruct seed j)
      언어가 같으므로 참 효과는 정의상 0이다. 여기서 "유의한 차이"가 나오면
      그것은 캘리브레이션 표본 변동이 만든 가짜 효과다. 이 비율이 위양성률이다.

추정량 (교차 짝 기준, 동률 분할 채점)
  L_KO = D(en_instruct, KMMLU) − D(ko, KMMLU)
  L_EN = D(en_instruct, MMLU)  − D(ko, MMLU)
  X    = L_KO − L_EN
  귀무 짝에서는 두 번째 모델을 "다른 조건"으로 간주해 같은 식을 쓴다.

캘리브레이션 크기별로 결과를 따로 낸다(RQ2: 크기가 커지면 위양성률이 줄어드는가).

사용법:
  python single_run_risk.py --base quant_experiments/EXAONE-3.5-7.8B-Instruct
  python single_run_risk.py --base ... --seeds random   # 결정적 추출(seed 0) 제외
"""

import argparse
import csv
import itertools
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paired_bootstrap import load_item_scores  # noqa: E402

BENCH = {"KMMLU": ("ko", "kmmlu_"), "MMLU": ("en", "mmlu_")}
DIR_RE = re.compile(r"^(gptq|awq)_(ko|en_instruct)(?:_n(\d+))?_s(\d+)$")
DEFAULT_N = 256


def discover(base):
    """(method, calib, n, seed) -> {lang: dir}"""
    out = defaultdict(dict)
    for lang in ("ko", "en"):
        root = base / "seed_eval" / lang
        if not root.is_dir():
            continue
        for d in root.iterdir():
            m = DIR_RE.match(d.name)
            if m and d.is_dir():
                n = int(m.group(3)) if m.group(3) else DEFAULT_N
                out[(m.group(1), m.group(2), n, int(m.group(4)))][lang] = d
    return out


def boot_acc(mat, iters, rng):
    """문항 재표집 정확도. 모든 모델이 같은 재표집을 공유한다(대응 구조). (iters, 모델수)"""
    n = mat.shape[0]
    out = np.empty((iters, mat.shape[1]))
    p = np.full(n, 1.0 / n)
    for i in range(iters):
        out[i] = (rng.multinomial(n, p) @ mat) / n
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--method", default="gptq", choices=["gptq", "awq"])
    ap.add_argument("--seeds", default="all", choices=["all", "random"],
                    help="random이면 결정적 추출(seed 0) 제외")
    ap.add_argument("--scoring", default="tie_split", choices=["tie_split", "first", "lm_eval"])
    ap.add_argument("--iters", type=int, default=1000)
    args = ap.parse_args()
    base = Path(args.base)
    rng = np.random.default_rng(2024)

    found = discover(base)
    sizes = sorted({k[2] for k in found if k[0] == args.method})
    if not sizes:
        print(f"{args.method} 결과가 없다.")
        return

    lines = [f"=== 단일 실행 위험도 ({args.method.upper()}, 채점 {args.scoring}, 시드 {args.seeds}) ===",
             "    귀무 짝 = 같은 언어 시드끼리의 짝. 참 효과가 0이므로 '유의'는 모두 가짜 효과다.",
             ""]
    rows = []
    summary = []

    for n in sizes:
        seeds = {c: sorted(s for (m, cc, nn, s), v in found.items()
                           if m == args.method and cc == c and nn == n
                           and "ko" in v and "en" in v and (args.seeds == "all" or s > 0))
                 for c in ("ko", "en_instruct")}
        if min(len(seeds["ko"]), len(seeds["en_instruct"])) < 2:
            lines.append(f"[n={n}] 시드가 조건당 2개 미만이라 건너뛴다: {seeds}")
            continue

        models = [("ko", s) for s in seeds["ko"]] + [("en_instruct", s) for s in seeds["en_instruct"]]
        acc_pt, acc_bs = {}, {}
        for bench, (lang, prefix) in BENCH.items():
            bf = load_item_scores(base / "seed_eval" / lang / "bf16", prefix, args.scoring)
            sc = [load_item_scores(found[(args.method, c, n, s)][lang], prefix, args.scoring)
                  for c, s in models]
            items = sorted(set(bf).intersection(*[set(x) for x in sc]))
            mat = np.array([[bf[it]] + [x[it] for x in sc] for it in items])
            acc_pt[bench] = mat.mean(0)
            acc_bs[bench] = boot_acc(mat, args.iters, rng)

        idx = {m: i + 1 for i, m in enumerate(models)}   # 0번 열은 BF16

        def D(acc, m):
            return (acc[..., 0] - acc[..., idx[m]]) / acc[..., 0] * 100

        def stats(a, b):
            """a를 기준 조건, b를 비교 조건으로 L_KO, L_EN, X의 점추정과 95% 구간"""
            res = {}
            for q in ("L_KO", "L_EN", "X"):
                vals = []
                for acc in (acc_pt, acc_bs):
                    lk = D(acc["KMMLU"], b) - D(acc["KMMLU"], a)
                    le = D(acc["MMLU"], b) - D(acc["MMLU"], a)
                    vals.append({"L_KO": lk, "L_EN": le, "X": lk - le}[q])
                lo, hi = np.quantile(vals[1], [0.025, 0.975])
                res[q] = (float(vals[0]), float(lo), float(hi), bool(lo > 0 or hi < 0))
            return res

        cross = [(("ko", i), ("en_instruct", j)) for i in seeds["ko"] for j in seeds["en_instruct"]]
        null = ([(("ko", i), ("ko", j)) for i, j in itertools.combinations(seeds["ko"], 2)] +
                [(("en_instruct", i), ("en_instruct", j))
                 for i, j in itertools.combinations(seeds["en_instruct"], 2)])

        lines.append(f"[n={n}] 시드 ko={seeds['ko']} en_instruct={seeds['en_instruct']}  "
                     f"교차 짝 {len(cross)}개, 귀무 짝 {len(null)}개")
        for kind, pairs in (("교차", cross), ("귀무", null)):
            res = [stats(a, b) for a, b in pairs]
            for (a, b), r in zip(pairs, res):
                for q, (pt, lo, hi, sig) in r.items():
                    rows.append([n, kind, f"{a[0]}_s{a[1]}", f"{b[0]}_s{b[1]}", q,
                                 round(pt, 3), round(lo, 3), round(hi, 3), int(sig)])
            for q in ("L_KO", "L_EN", "X"):
                pts = np.array([r[q][0] for r in res])
                sig = np.array([r[q][3] for r in res])
                if kind == "교차":
                    lines.append(f"    교차 {q:4s} 유의 {sig.mean() * 100:5.1f}%  "
                                 f"양수 {np.mean(pts > 0) * 100:5.1f}%  음수 {np.mean(pts < 0) * 100:5.1f}%  "
                                 f"점추정 범위 [{pts.min():+.2f}, {pts.max():+.2f}]")
                else:
                    lines.append(f"    귀무 {q:4s} 위양성률 {sig.mean() * 100:5.1f}%  "
                                 f"(|점추정| 최대 {np.abs(pts).max():.2f}%p)")
                    if q == "X":
                        summary.append((n, float(sig.mean()), len(null)))

        # 시드 간 변동 대 문항 변동
        for c in ("ko", "en_instruct"):
            for bench in BENCH:
                d_pt = np.array([D(acc_pt[bench], (c, s)) for s in seeds[c]])
                d_se = np.mean([np.std(D(acc_bs[bench], (c, s)), ddof=1) for s in seeds[c]])
                lines.append(f"    D[{c:11s}] {bench:5s} 시드간 SD {d_pt.std(ddof=1):.2f}%p  "
                             f"문항 SE {d_se:.2f}%p  비율 {d_pt.std(ddof=1) / d_se:.2f}")
        lines.append("")

    if summary:
        lines.append("=== RQ2 요약: 캘리브레이션 크기별 X의 위양성률 (귀무 짝) ===")
        for n, fpr, k in summary:
            lines.append(f"  n={n:5d}  위양성률 {fpr * 100:5.1f}%  (귀무 짝 {k}개)")
        lines.append("  임계점 정의(사전 고정): 위양성률이 5% 이하로 떨어지는 최소 n")

    out = base / "risk"
    out.mkdir(parents=True, exist_ok=True)
    tag = f"{args.method}_{args.scoring}_{args.seeds}"
    text = "\n".join(lines)
    (out / f"single_run_risk_{tag}.txt").write_text(text + "\n", encoding="utf-8")
    with open(out / f"single_run_risk_{tag}.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["n", "pair_type", "cond_a", "cond_b", "quantity", "point", "ci_low", "ci_high", "significant"])
        w.writerows(rows)
    print(text)
    print(f"\n저장 → {out}")


if __name__ == "__main__":
    main()
