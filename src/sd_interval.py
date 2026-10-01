#!/usr/bin/env python3
"""
sd_interval.py — 표준편차 자체의 불확실성과 두 규모 사이의 비교

왜 필요한가.

본 논문은 SD(X)를 핵심 수치로 쓴다. 그런데 7.8B는 18시드, 32B는 6시드다.
6개 표본에서 얻은 SD는 그 자체가 매우 부정확하다 — 카이제곱 분포에 따라
참값이 점추정치의 0.62배에서 2.45배 사이에 있을 수 있다. 이것을 밝히지 않고
"32B의 SD는 7.8B와 비슷하다/다르다"고 쓰면, 단일 실행 결과를 믿지 말라는
논문이 정확히 같은 과오를 범한다.

이 스크립트는 세 가지를 계산한다.
  1. SD의 신뢰구간 (카이제곱, 정규 가정)
     [s*sqrt(df/chi2_{0.975}), s*sqrt(df/chi2_{0.025})]
  2. 부트스트랩 SD 구간 (정규 가정 없음). 표본이 적으면 이 구간도 좁게 나오는
     경향이 있으므로 카이제곱 구간과 함께 본다.
  3. 두 조건 비교: SD 구간의 겹침 여부, 부호 반전 비율.
     겹치면 "두 규모의 변동 크기를 구분할 수 없다"가 정확한 서술이다.
     F검정을 쓰지 않는 이유는 n=6에서 정규 가정에 크게 의존하기 때문이며,
     구간 겹침으로 판단하는 편이 보수적이고 해석이 분명하다.

사용법
  # 시드별 X 값을 직접 넘긴다
  python sd_interval.py --a 3.1 0.4 -0.8 2.2 1.5 ... --label-a "7.8B (18시드)" \\
                        --b 1.9 2.8 -0.5 1.2 0.9 1.4 --label-b "32B (6시드)"

  # 또는 JSON 파일에서 (키가 라벨, 값이 리스트)
  python sd_interval.py --json seedwise_X.json
"""

import argparse
import json
import math
import sys
from typing import List, Optional, Sequence

import numpy as np

# 카이제곱 분위수 (양측 95%). 표준 표값이며 scipy 의존을 피하기 위해 내장한다.
CHI2 = {
    1: (0.000982, 5.024), 2: (0.0506, 7.378), 3: (0.216, 9.348),
    4: (0.484, 11.143), 5: (0.831, 12.833), 6: (1.237, 14.449),
    7: (1.690, 16.013), 8: (2.180, 17.535), 9: (2.700, 19.023),
    10: (3.247, 20.483), 11: (3.816, 21.920), 12: (4.404, 23.337),
    13: (5.009, 24.736), 14: (5.629, 26.119), 15: (6.262, 27.488),
    16: (6.908, 28.845), 17: (7.564, 30.191), 18: (8.231, 31.526),
    19: (8.907, 32.852), 20: (9.591, 34.170), 24: (12.401, 39.364),
    29: (16.047, 45.722), 39: (23.654, 58.120),
}


def chi2_pair(df: int):
    if df in CHI2:
        return CHI2[df]
    # 표에 없는 자유도는 Wilson–Hilferty 근사로 보간한다(df가 클 때 정확하다).
    z = 1.959964
    lo = df * (1 - 2 / (9 * df) + z * math.sqrt(2 / (9 * df))) ** 3
    hi = df * (1 - 2 / (9 * df) - z * math.sqrt(2 / (9 * df))) ** 3
    return (min(lo, hi), max(lo, hi))


def sd_chi2_ci(x: Sequence[float]):
    n = len(x)
    if n < 2:
        return (float("nan"),) * 3
    s = float(np.std(x, ddof=1))
    df = n - 1
    q_lo, q_hi = chi2_pair(df)
    return s, s * math.sqrt(df / q_hi), s * math.sqrt(df / q_lo)


def boot(x: Sequence[float], stat, reps: int = 20000, seed: int = 0):
    rng = np.random.default_rng(seed)
    a = np.asarray(x, dtype=float)
    n = len(a)
    vals = np.empty(reps)
    for i in range(reps):
        vals[i] = stat(a[rng.integers(0, n, n)])
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def describe(label: str, x: Sequence[float]) -> dict:
    n = len(x)
    a = np.asarray(x, dtype=float)
    s, lo, hi = sd_chi2_ci(a)
    b_lo, b_hi = boot(a, lambda v: np.std(v, ddof=1))
    m = float(a.mean())
    se = s / math.sqrt(n)
    m_lo, m_hi = boot(a, np.mean)
    neg = int((a < 0).sum())
    df = n - 1
    q_lo, q_hi = chi2_pair(df)

    print(f"### {label}  (n={n})")
    print(f"  평균 X            {m:+.3f}   부트스트랩 95% [{m_lo:+.3f}, {m_hi:+.3f}]")
    print(f"                    (캘리브레이션 표본 변동만 반영한 구간이다.")
    print(f"                     평가 문항 변동까지 합친 두 단계 구간은 final_analysis.py가 낸다)")
    print(f"  SD(X)             {s:.3f}")
    print(f"    카이제곱 95%    [{lo:.3f}, {hi:.3f}]   (배율 {lo/s:.2f}x ~ {hi/s:.2f}x, df={df})")
    print(f"    부트스트랩 95%  [{b_lo:.3f}, {b_hi:.3f}]")
    print(f"  SE(평균)          {se:.3f}")
    print(f"  부호 반전         {neg}/{n} ({100*neg/n:.1f}%)")
    if n < 10:
        print(f"  주의: n={n}에서 SD의 참값은 점추정치의 {lo/s:.2f}~{hi/s:.2f}배 범위에 있다.")
        print(f"        이 조건의 SD는 '크기'를 말하기에 부족하고, 부호와 구조를 말하는 데 쓰는 것이 맞다.")
    print()
    return {"label": label, "n": n, "mean": m, "mean_ci": [m_lo, m_hi],
            "sd": s, "sd_chi2_ci": [lo, hi], "sd_boot_ci": [b_lo, b_hi],
            "se_mean": se, "sign_reversals": neg}


def compare(A: dict, B: dict, xa: Sequence[float], xb: Sequence[float],
            seed: int = 0) -> None:
    print("### 두 조건 비교")
    la, ha = A["sd_chi2_ci"]
    lb, hb = B["sd_chi2_ci"]
    overlap = not (ha < lb or hb < la)
    print(f"  SD 구간   {A['label']}: [{la:.3f}, {ha:.3f}]")
    print(f"            {B['label']}: [{lb:.3f}, {hb:.3f}]")
    print(f"  겹침      {'예' if overlap else '아니오'}")
    if overlap:
        print("  -> 두 조건의 변동 크기를 이 데이터로 구분할 수 없다. 논문에는")
        print("     'SD가 비슷하다'가 아니라 '구분할 수 없다'로 써야 한다.")
    else:
        print("  -> 변동 크기가 다르다고 말할 수 있다. 어느 쪽이 큰지 방향까지 함께 쓸 것.")

    # SD 비의 부트스트랩 구간 (두 표본을 독립적으로 재표집)
    rng = np.random.default_rng(seed)
    a, b = np.asarray(xa, float), np.asarray(xb, float)
    reps = 20000
    r = np.empty(reps)
    for i in range(reps):
        sa = np.std(a[rng.integers(0, len(a), len(a))], ddof=1)
        sb = np.std(b[rng.integers(0, len(b), len(b))], ddof=1)
        r[i] = sa / sb if sb > 0 else np.nan
    r = r[~np.isnan(r)]
    print(f"  SD 비 ({A['label'].split()[0]}/{B['label'].split()[0]}) "
          f"{A['sd']/B['sd']:.3f}  부트스트랩 95% "
          f"[{np.percentile(r, 2.5):.3f}, {np.percentile(r, 97.5):.3f}]")
    print("  (1을 포함하면 크기 차이를 주장할 수 없다)")

    # 부호 반전 비율
    pa = A["sign_reversals"] / A["n"]
    pb = B["sign_reversals"] / B["n"]
    print(f"  부호 반전 비율   {pa*100:.1f}% vs {pb*100:.1f}%")
    if min(A["n"], B["n"]) < 10:
        print("  (표본이 적은 쪽의 비율은 한 건이 바뀌면 크게 흔들린다. 비율 비교는")
        print("   '같은 방향으로 관측된다' 수준까지만 주장할 것)")
    print()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", nargs="+", type=float, help="조건 A의 시드별 X 값")
    ap.add_argument("--b", nargs="+", type=float, help="조건 B의 시드별 X 값")
    ap.add_argument("--label-a", default="조건 A")
    ap.add_argument("--label-b", default="조건 B")
    ap.add_argument("--json", help='{"라벨": [값...], ...} 형식 파일')
    ap.add_argument("--out", help="결과 JSON 저장 경로")
    ap.add_argument("--seed", type=int, default=0)
    g = ap.parse_args()

    sets = []
    if g.json:
        data = json.loads(open(g.json, encoding="utf-8").read())
        for k, v in data.items():
            sets.append((k, list(map(float, v))))
    else:
        if not g.a:
            ap.error("--a 또는 --json 중 하나가 필요하다")
        sets.append((g.label_a, g.a))
        if g.b:
            sets.append((g.label_b, g.b))

    print("=" * 66)
    print("SD(X)의 불확실성 — 표본이 적을 때 SD를 어디까지 말할 수 있는가")
    print("=" * 66)
    print()
    out = [describe(lab, v) for lab, v in sets]
    if len(sets) == 2:
        compare(out[0], out[1], sets[0][1], sets[1][1], seed=g.seed)

    if g.out:
        with open(g.out, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"저장 → {g.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
