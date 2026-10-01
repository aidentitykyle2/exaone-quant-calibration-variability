#!/usr/bin/env python3
"""
permutation_test.py — 캘리브레이션 언어 라벨에 대한 순열 검정

왜 필요한가.

짝 단위 분석(귀무 짝 30개 중 몇 개가 0을 배제하는가)은 같은 시드가 여러 짝에
반복 등장하므로 독립 반복이 아니다. 6개 시드에서 만든 15개 짝을 15번의 독립
실험처럼 세면 비율의 해석이 어긋난다. 짝 세기는 탐색적 시각화로 두고, 추론은
**모델 라벨을 재배정하는 순열 검정**으로 한다.

절차.
  각 양자화 모델은 (KMMLU 저하율, MMLU 저하율) 한 쌍을 갖는다. 어느 모델이
  한국어 캘리브레이션이고 어느 모델이 영어 캘리브레이션인지를 무작위로 다시
  배정한 뒤 같은 통계량을 계산한다. 두 벤치마크를 함께 옮기므로 짝 재사용에서
  오는 의존성이 생기지 않는다. 캘리브레이션 언어가 효과가 없다는 귀무가설
  아래에서 라벨은 교환 가능하므로, 이 재배정이 정확한 귀무분포를 만든다.

  Δ_KO = mean_en(D_KMMLU) − mean_ko(D_KMMLU)     (양수 예상)
  Δ_EN = mean_ko(D_MMLU)  − mean_en(D_MMLU)      (양수 예상)
  X    = Δ_KO + Δ_EN

  조건당 시드가 6개면 C(12,6)=924가지를 전수 열거하므로 근사가 없다.
  18개면 C(36,18)이 90억을 넘어 몬테카를로로 추정한다.

무엇을 보고하는가.
  p값뿐 아니라 **귀무분포의 산포와 분위수**를 함께 낸다. 관측값이 유의하더라도
  귀무분포가 비슷한 크기까지 도달한다면, 단일 실행에서 그 크기를 보았을 때
  언어 효과로 귀속할 수 없다는 본 연구의 논지가 검정 형태로 확인된다.

사용법
  python permutation_test.py --base quant_experiments/EXAONE-3.5-32B-Instruct
  python permutation_test.py --base ... --scoring first --iters 200000
  python permutation_test.py --base ... --out perm.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

EXACT_LIMIT = 200_000     # 전수 열거를 시도할 최대 조합 수


def load(base: Path, scoring: str, ko_task: str, en_task: str,
         method: str) -> dict:
    """rescore_long.csv에서 조건별 (한국어 저하율, 영어 저하율)을 모은다."""
    f = base / "rescore" / "rescore_long.csv"
    if not f.exists():
        raise SystemExit(f"{f} 가 없다. rescore_all.py를 먼저 실행할 것.")
    deg = {}
    with f.open(encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            if r["scoring"] != scoring:
                continue
            m = r["model"]
            if not m.startswith(method + "_") or m == "bf16":
                continue
            task = r["task"]
            if task not in (ko_task, en_task):
                continue
            try:
                d = float(r["degradation_pct"])
            except (TypeError, ValueError):
                continue
            deg.setdefault(m, {})[task] = d

    out = {}
    for m, v in deg.items():
        if ko_task in v and en_task in v:
            # 이름 규칙: <method>_<calib>_s<seed>
            body = m[len(method) + 1:]
            calib, _, seed = body.rpartition("_s")
            if calib and seed.isdigit():
                out[m] = (calib, int(seed), v[ko_task], v[en_task])
    return out


def statistic(M: np.ndarray, ko_idx: np.ndarray, en_idx: np.ndarray):
    d_ko = M[en_idx, 0].mean() - M[ko_idx, 0].mean()
    d_en = M[ko_idx, 1].mean() - M[en_idx, 1].mean()
    return d_ko, d_en, d_ko + d_en


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--scoring", default="tie_split",
                    choices=["tie_split", "first", "lm_eval"])
    ap.add_argument("--method", default="gptq")
    ap.add_argument("--ko-task", default="kmmlu")
    ap.add_argument("--en-task", default="mmlu")
    ap.add_argument("--ko-calib", default="ko")
    ap.add_argument("--en-calib", default="en_instruct")
    ap.add_argument("--iters", type=int, default=200_000,
                    help="전수 열거가 불가능할 때의 몬테카를로 반복 수")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    base = Path(a.base)
    rows = load(base, a.scoring, a.ko_task, a.en_task, a.method)
    ko = sorted((s, k, e) for c, s, k, e in rows.values() if c == a.ko_calib)
    en = sorted((s, k, e) for c, s, k, e in rows.values() if c == a.en_calib)
    if len(ko) < 2 or len(en) < 2:
        raise SystemExit(f"조건별 시드가 부족하다 (ko {len(ko)}, en {len(en)}). "
                         f"--ko-calib / --en-calib 값을 확인할 것.")
    if len(ko) != len(en):
        print(f"주의: 조건별 시드 수가 다르다 (ko {len(ko)}, en {len(en)}). "
              f"라벨 교환은 그대로 유효하나 해석에 유의할 것.")

    M = np.array([[k, e] for _, k, e in ko] + [[k, e] for _, k, e in en], dtype=float)
    n, n_ko = len(M), len(ko)
    obs_ko = np.arange(n_ko)
    obs_en = np.arange(n_ko, n)
    obs = statistic(M, obs_ko, obs_en)

    total = math.comb(n, n_ko)
    exact = total <= EXACT_LIMIT

    print("=" * 70)
    print(f"순열 검정 — 캘리브레이션 언어 라벨 재배정 ({base.name})")
    print("=" * 70)
    print(f"채점 {a.scoring} · 기법 {a.method} · 시드 {a.ko_calib} {len(ko)}개 / "
          f"{a.en_calib} {len(en)}개")
    print(f"평가 {a.ko_task}(한국어) / {a.en_task}(영어)")
    print(f"가능한 라벨 배정 {total:,}가지 → "
          f"{'전수 열거 (근사 없음)' if exact else f'몬테카를로 {a.iters:,}회'}")
    print()
    print(f"관측값   Δ_KO {obs[0]:+.4f}   Δ_EN {obs[1]:+.4f}   X {obs[2]:+.4f}")
    print()

    if exact:
        idx = np.array(list(combinations(range(n), n_ko)))
        vals = np.empty((len(idx), 3))
        allset = np.arange(n)
        for i, c in enumerate(idx):
            mask = np.zeros(n, dtype=bool); mask[c] = True
            vals[i] = statistic(M, allset[mask], allset[~mask])
    else:
        rng = np.random.default_rng(a.seed)
        vals = np.empty((a.iters, 3))
        allset = np.arange(n)
        for i in range(a.iters):
            perm = rng.permutation(n)
            vals[i] = statistic(M, perm[:n_ko], perm[n_ko:])

    names = ["Δ_KO", "Δ_EN", "X   "]
    res = {}
    print(f"  {'지표':6s} {'관측':>9s} {'귀무 SD':>9s} {'양측 p':>9s} {'단측 p':>9s}")
    for k, nm in enumerate(names):
        v, ob = vals[:, k], obs[k]
        # 전수 열거에서는 관측 배정 자체가 포함되므로 그대로, 몬테카를로는 +1 보정
        p2 = float((np.abs(v) >= abs(ob) - 1e-12).mean())
        p1 = float((v >= ob - 1e-12).mean())
        if not exact:
            p2 = (p2 * a.iters + 1) / (a.iters + 1)
            p1 = (p1 * a.iters + 1) / (a.iters + 1)
        print(f"  {nm:6s} {ob:+9.4f} {v.std(ddof=1):9.4f} {p2:9.4f} {p1:9.4f}")
        res[nm.strip()] = {"observed": float(ob), "null_sd": float(v.std(ddof=1)),
                           "p_two_sided": p2, "p_one_sided": p1}

    print()
    print("  귀무분포가 관측값 크기에 얼마나 도달하는가")
    print(f"  {'지표':6s} {'관측 |값|':>10s} {'귀무 95분위':>12s} {'귀무 최대':>10s} {'판정':>26s}")
    for k, nm in enumerate(names):
        av = np.abs(vals[:, k]); ob = abs(obs[k])
        q95, mx = float(np.percentile(av, 95)), float(av.max())
        verdict = "귀무가 관측 크기를 흔히 낸다" if q95 >= ob else "관측이 귀무 상위 5%를 넘는다"
        print(f"  {nm:6s} {ob:10.3f} {q95:12.3f} {mx:10.3f} {verdict:>26s}")
        res[nm.strip()].update({"null_p95_abs": q95, "null_max_abs": mx})

    print()
    print("  해석 지침")
    print("   · p값은 '관측된 차이를 캘리브레이션 언어에 귀속할 수 있는가'에 답한다.")
    print("   · 귀무 95분위가 관측값을 넘으면, 그 크기의 값은 라벨을 무작위로 바꿔도")
    print("     흔히 나오므로 단일 실행에서 관측했을 때 언어 효과로 읽을 수 없다.")
    print("   · 라벨 교환 가능성이 귀무가설이므로 시드 재사용에 따른 의존성이 없다.")

    if a.out:
        Path(a.out).write_text(json.dumps({
            "base": str(base), "scoring": a.scoring, "method": a.method,
            "n_ko": len(ko), "n_en": len(en), "exact": exact,
            "n_assignments": total, "iters": None if exact else a.iters,
            "results": res,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n저장 → {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
