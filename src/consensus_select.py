#!/usr/bin/env python3
"""
consensus_select.py — 다중 표본 합의 캘리브레이션 평가 (E3, RQ3)

같은 총 표본 예산에서 세 전략의 안정성을 비교한다.
  S1 단일      n=256 한 번 양자화 (기준)
  C1 통합      n=1024 한 번 양자화 (256 × 4와 같은 예산)
  C2 선택      n=256으로 K번 양자화한 뒤 개발용 문항에서 가장 좋은 모델을 고른다

C2의 선택 기준
  agree  BF16과의 판정 일치도. 정답 라벨이 필요 없어 실무에서 바로 쓸 수 있다.
  acc    개발 문항 정확도. 라벨이 있는 소규모 검증 세트를 가정한다.

선택 편향 방지
  각 문항을 (과제, doc_id)의 해시로 개발 절반과 시험 절반에 고정 배정한다.
  선택은 개발 절반으로만 하고, 모든 전략의 성능은 시험 절반으로만 잰다.

주 지표: 시험 절반에서의 언어 효과 X 추정치가 반복에 따라 얼마나 흔들리는가(SD).
  S1·C1은 가능한 모든 (ko 시드, en_instruct 시드) 짝, C2는 가능한 모든 K-부분집합
  조합에서 X를 구한다. 합의 캘리브레이션이 단일 실행의 위험을 줄인다면 SD가 작아야 한다.

한계: n=256 시드가 적으면 C2의 K-부분집합들이 서로 크게 겹쳐 SD가 과소 추정된다.
      시드 수를 함께 보고한다.

사용법:
  python consensus_select.py --base quant_experiments/EXAONE-3.5-7.8B-Instruct --k 3
"""

import argparse
import hashlib
import itertools
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paired_bootstrap import target_index  # noqa: E402
from single_run_risk import discover  # noqa: E402

SAMPLE_RE = re.compile(r"samples_(.+?)_(\d{4}-\d{2}-\d{2}T[\d\-.]+)\.jsonl$")
BENCH = {"KMMLU": ("ko", "kmmlu_"), "MMLU": ("en", "mmlu_")}


def load(model_dir, prefix):
    """(task, doc_id) -> (최대값 동률 집합, 정답 번호)"""
    latest = {}
    for p in model_dir.rglob("samples_*.jsonl"):
        m = SAMPLE_RE.search(p.name)
        if m and m.group(1).startswith(prefix):
            if m.group(1) not in latest or m.group(2) > latest[m.group(1)][0]:
                latest[m.group(1)] = (m.group(2), p)
    out = {}
    for task, (_, p) in latest.items():
        for line in open(p, encoding="utf-8"):
            r = json.loads(line)
            fr, t = r.get("filtered_resps"), target_index(r)
            if not fr or t is None:
                continue
            ll = np.array([float(x[0]) if isinstance(x, (list, tuple)) else float(x) for x in fr])
            out[(task, r["doc_id"])] = (frozenset(np.flatnonzero(ll == ll.max()).tolist()), t)
    return out


def is_dev(key):
    return int(hashlib.md5(f"{key[0]}|{key[1]}".encode()).hexdigest(), 16) % 2 == 0


def acc(rec, items):
    return float(np.mean([(1.0 if rec[i][1] in rec[i][0] else 0.0) / len(rec[i][0]) for i in items]))


def agree(ref, rec, items):
    return float(np.mean([len(ref[i][0] & rec[i][0]) / (len(ref[i][0]) * len(rec[i][0])) for i in items]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--method", default="gptq")
    ap.add_argument("--k", type=int, default=3, help="C2에서 양자화할 횟수")
    ap.add_argument("--n-small", type=int, default=256)
    ap.add_argument("--n-pooled", type=int, default=1024)
    ap.add_argument("--seeds", default="random", choices=["all", "random"])
    args = ap.parse_args()
    base = Path(args.base)
    found = discover(base)

    def seeds_for(c, n):
        return sorted(s for (m, cc, nn, s), v in found.items()
                      if m == args.method and cc == c and nn == n and "ko" in v and "en" in v
                      and (args.seeds == "all" or s > 0))

    ref, split, data = {}, {}, {}
    for bench, (lang, prefix) in BENCH.items():
        ref[bench] = load(base / "seed_eval" / lang / "bf16", prefix)
        keys = sorted(ref[bench])
        split[bench] = ([k for k in keys if is_dev(k)], [k for k in keys if not is_dev(k)])
    for (m, c, n, s), v in found.items():
        if m == args.method and n in (args.n_small, args.n_pooled) and "ko" in v and "en" in v:
            if args.seeds == "random" and s == 0:
                continue
            data[(c, n, s)] = {b: load(v[BENCH[b][0]], BENCH[b][1]) for b in BENCH}

    def test_D(c, n, s):
        out = {}
        for b in BENCH:
            items = [i for i in split[b][1] if i in data[(c, n, s)][b]]
            bf, q = acc(ref[b], items), acc(data[(c, n, s)][b], items)
            out[b] = (bf - q) / bf * 100
        return out

    def dev_score(c, n, s, crit):
        vals = []
        for b in BENCH:
            items = [i for i in split[b][0] if i in data[(c, n, s)][b]]
            vals.append(agree(ref[b], data[(c, n, s)][b], items) if crit == "agree"
                        else acc(data[(c, n, s)][b], items))
        return float(np.mean(vals))

    def X_of(dk, de):
        return (de["KMMLU"] - dk["KMMLU"]) - (de["MMLU"] - dk["MMLU"])

    lines = [f"=== 다중 표본 합의 캘리브레이션 ({args.method.upper()}, 시드 {args.seeds}) ===",
             f"    개발/시험 문항 수: KMMLU {len(split['KMMLU'][0])}/{len(split['KMMLU'][1])}, "
             f"MMLU {len(split['MMLU'][0])}/{len(split['MMLU'][1])}", ""]
    results = {}

    for label, n in (("S1 단일", args.n_small), ("C1 통합", args.n_pooled)):
        sk, se = seeds_for("ko", n), seeds_for("en_instruct", n)
        if not sk or not se:
            lines.append(f"{label} (n={n}): 시드 없음")
            continue
        Dk = {s: test_D("ko", n, s) for s in sk}
        De = {s: test_D("en_instruct", n, s) for s in se}
        results[label] = ([X_of(Dk[i], De[j]) for i in sk for j in se], sk, se, Dk, De)

    sk, se = seeds_for("ko", args.n_small), seeds_for("en_instruct", args.n_small)
    if len(sk) >= args.k and len(se) >= args.k:
        for crit in ("agree", "acc"):
            pick = {}
            for c, ss in (("ko", sk), ("en_instruct", se)):
                pick[c] = [max(combo, key=lambda s: dev_score(c, args.n_small, s, crit))
                           for combo in itertools.combinations(ss, args.k)]
            Dk = {s: test_D("ko", args.n_small, s) for s in set(pick["ko"])}
            De = {s: test_D("en_instruct", args.n_small, s) for s in set(pick["en_instruct"])}
            xs = [X_of(Dk[a], De[b]) for a in pick["ko"] for b in pick["en_instruct"]]
            results[f"C2 선택({crit}, K={args.k})"] = (xs, pick["ko"], pick["en_instruct"], Dk, De)
    else:
        lines.append(f"C2: n={args.n_small} 시드가 K={args.k}보다 적다 (ko {len(sk)}, en_instruct {len(se)})")

    lines.append(f"{'전략':26s} {'X 평균':>8s} {'X SD':>7s} {'X 범위':>18s}  {'D_KO[ko] SD':>11s} {'D_EN[en_i] SD':>13s}  반복 수")
    for label, (xs, pk, pe, Dk, De) in results.items():
        xs = np.array(xs)
        dko = np.array([Dk[s]["KMMLU"] for s in (pk if isinstance(pk, list) else [])])
        den = np.array([De[s]["MMLU"] for s in (pe if isinstance(pe, list) else [])])
        sd = lambda a: a.std(ddof=1) if len(a) > 1 else float("nan")
        lines.append(f"{label:26s} {xs.mean():+8.2f} {sd(xs):7.2f} [{xs.min():+6.2f}, {xs.max():+6.2f}]  "
                     f"{sd(dko):11.2f} {sd(den):13.2f}  {len(xs)}")
    lines.append("")
    lines.append("해석: 같은 예산에서 C1·C2의 X SD가 S1보다 뚜렷이 작으면 합의 캘리브레이션이")
    lines.append("      단일 실행의 위험을 줄인다는 근거가 된다. C2(agree)가 C2(acc)와 비슷하면")
    lines.append("      라벨 없이도 같은 효과를 얻을 수 있다는 뜻이다.")
    lines.append("주의: C2의 반복들은 같은 시드를 공유하는 부분집합이라 서로 독립이 아니며 SD가")
    lines.append("      과소 추정될 수 있다. n=256 시드 수를 늘릴수록 신뢰도가 높아진다.")

    out = base / "consensus"
    out.mkdir(parents=True, exist_ok=True)
    text = "\n".join(lines)
    (out / f"consensus_{args.method}_k{args.k}.txt").write_text(text + "\n", encoding="utf-8")
    print(text)
    print(f"\n저장 → {out}")


if __name__ == "__main__":
    main()
