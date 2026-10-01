#!/usr/bin/env python3
"""
robust_selftest.py — robust_gptq.aggregate_hessians()의 자체 검증

*** 향후 연구용 코드다. 본 논문의 결과에는 사용되지 않았다. ***

GPU도 llmcompressor도 필요 없다(torch만 있으면 된다). 강건 집계를 실제 32B에
돌리기 전에, 집계 수식이 의도대로 동작하는지를 먼저 여기서 확인한다.
본 논문이 "단일 실행 결과를 믿지 말라"는 주장을 하는 만큼, 완화 절차 자체도
검증 없이 결과를 내놓으면 같은 비판을 받는다.

점검 항목
  T1 mean == 풀링       부분집합으로 나눈 뒤 mean으로 합치면 나누지 않은
                        H/n 과 수치적으로 같아야 한다. 이게 틀리면 배선 버그다.
                        (부분집합 크기가 불균등할 때도 성립해야 한다)
  T2 K=1 항등           K=1이면 어떤 집계 방식이든 H/n 이어야 한다.
  T3 PSD·대칭 보존      geomedian / trimmed / mean 결과가 대칭이고 최소
                        고유값이 -tol 이상이어야 한다(볼록결합의 성질).
  T4 강건성             부분집합 하나를 오염시켰을 때(활성화 크기 폭증)
                        mean은 크게 끌려가고 geomedian·trimmed는 덜 끌려가야
                        한다. 이 차이가 없으면 완화 가설의 전제가 무너진다.
  T5 블록 분할 무영향   같은 표본을 K개로 어떻게 쪼개도 mean 결과는 동일해야 한다.
  T6 median 경고        median은 PSD가 보장되지 않음을 실제로 보여준다
                        (실패가 아니라 기록 — 주 결과로 쓰지 말아야 하는 근거).

사용법:
  python robust_selftest.py
  python robust_selftest.py --dim 256 --samples 512 --subsets 4
"""

import argparse
import math
import sys

import torch

from robust_gptq import aggregate_hessians

GPTQ_PRECISION = torch.float32


def make_hessian(X: torch.Tensor) -> torch.Tensor:
    """llmcompressor의 accumulate_hessian과 같은 형태: H = 2 * X^T X, n = 행 수."""
    Xs = math.sqrt(2) * X.to(GPTQ_PRECISION)
    return Xs.t().matmul(Xs)


def split_sizes(n: int, k: int, uneven: bool = False):
    if not uneven:
        per = n // k
        sizes = [per] * k
        sizes[-1] += n - per * k
        return sizes
    # 의도적으로 불균등하게: 1:2:3:... 비율
    w = list(range(1, k + 1))
    tot = sum(w)
    sizes = [max(1, n * wi // tot) for wi in w]
    sizes[-1] += n - sum(sizes)
    return sizes


def subsets_from(X: torch.Tensor, sizes):
    Hs, ns, off = [], [], 0
    for s in sizes:
        blk = X[off:off + s]
        Hs.append(make_hessian(blk))
        ns.append(float(blk.shape[0]))
        off += s
    return Hs, ns


def rel_err(a: torch.Tensor, b: torch.Tensor) -> float:
    denom = max(float(b.abs().max()), 1e-30)
    return float((a - b).abs().max()) / denom


def min_eig(M: torch.Tensor) -> float:
    return float(torch.linalg.eigvalsh(M.to(torch.float64)).min())


def asym(M: torch.Tensor) -> float:
    denom = max(float(M.abs().max()), 1e-30)
    return float((M - M.t()).abs().max()) / denom


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, default=128, help="가상 입력 차원(Hessian 한 변)")
    ap.add_argument("--samples", type=int, default=256, help="가상 캘리브레이션 표본 수")
    ap.add_argument("--subsets", type=int, default=4, help="부분집합 수 K")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    d, n, K = a.dim, a.samples, a.subsets

    # 활성화를 흉내 낸 입력: 채널별 스케일 편차 + 헤비테일(일부 문장이 큰 활성화를
    # 유발하는 상황). 실제 LLM 활성화의 outlier 구조를 단순 모사한 것이다.
    scale = torch.rand(d) * 3.0 + 0.2
    X = torch.randn(n, d) * scale
    heavy = torch.rand(n) < 0.05
    X[heavy] *= 6.0

    H_all = make_hessian(X)
    ref = H_all / float(n)                      # 기존 GPTQ가 쓰는 값
    tol = 1e-5
    fails = []

    print(f"설정: dim={d} samples={n} K={K} seed={a.seed}")
    print("-" * 62)

    # ---- T1: mean == 풀링 (균등/불균등 분할 모두)
    for label, uneven in (("균등", False), ("불균등", True)):
        sizes = split_sizes(n, K, uneven)
        Hs, ns = subsets_from(X, sizes)
        got = aggregate_hessians(Hs, ns, aggregation="mean")
        e = rel_err(got, ref)
        ok = e < tol
        print(f"T1 mean == 풀링 ({label} {sizes}): 상대오차 {e:.3e}  {'통과' if ok else '실패'}")
        if not ok:
            fails.append(f"T1-{label}")

    # ---- T2: K=1 항등
    for agg in ("mean", "median", "trimmed", "geomedian"):
        got = aggregate_hessians([H_all], [float(n)], aggregation=agg, trim=1)
        e = rel_err(got, ref)
        ok = e < tol
        if not ok:
            fails.append(f"T2-{agg}")
        print(f"T2 K=1 항등 ({agg:9s}): 상대오차 {e:.3e}  {'통과' if ok else '실패'}")

    # ---- T3: PSD·대칭 (볼록결합 계열)
    sizes = split_sizes(n, K)
    Hs, ns = subsets_from(X, sizes)
    ref_scale = float(ref.abs().max())
    for agg in ("mean", "trimmed", "geomedian"):
        got = aggregate_hessians(Hs, ns, aggregation=agg, trim=1)
        me, sy = min_eig(got), asym(got)
        ok = (me > -1e-6 * ref_scale) and (sy < 1e-6)
        if not ok:
            fails.append(f"T3-{agg}")
        print(f"T3 PSD·대칭 ({agg:9s}): 최소고유값 {me:+.3e} 비대칭 {sy:.1e}  "
              f"{'통과' if ok else '실패'}")

    # ---- T4: 강건성 — 부분집합 하나만 오염
    Xc = X.clone()
    bad_from = sum(sizes[:-1])
    Xc[bad_from:] *= 20.0                       # 마지막 부분집합의 활성화를 20배로
    Hs_c, ns_c = subsets_from(Xc, sizes)
    print("-" * 62)
    print("T4 강건성 (부분집합 1/K 오염, 기준=오염 전 풀링값과의 상대오차)")
    drift = {}
    for agg in ("mean", "median", "trimmed", "geomedian"):
        got = aggregate_hessians(Hs_c, ns_c, aggregation=agg, trim=1)
        drift[agg] = rel_err(got, ref)
        print(f"   {agg:9s}: {drift[agg]:.4f}")
    ok = (drift["geomedian"] < 0.5 * drift["mean"]) and (drift["trimmed"] < drift["mean"])
    if not ok:
        fails.append("T4")
    print(f"   geomedian < 0.5 * mean 이고 trimmed < mean : {'통과' if ok else '실패'}")

    # ---- T5: 분할 방식 무영향 (mean)
    print("-" * 62)
    base = aggregate_hessians(*subsets_from(X, split_sizes(n, K)), aggregation="mean")
    worst = 0.0
    for kk in (2, 3, K, K * 2):
        if kk > n:
            continue
        g = aggregate_hessians(*subsets_from(X, split_sizes(n, kk)), aggregation="mean")
        worst = max(worst, rel_err(g, base))
    ok = worst < tol
    if not ok:
        fails.append("T5")
    print(f"T5 분할 수 무영향 (mean, K=2/3/{K}/{K*2}): 최대 상대오차 {worst:.3e}  "
          f"{'통과' if ok else '실패'}")

    # ---- T6: median의 PSD 미보장 (기록용, 실패로 세지 않음)
    gm = aggregate_hessians(Hs_c, ns_c, aggregation="median")
    gp = aggregate_hessians(Hs_c, ns_c, aggregation="median", psd_project=True)
    print(f"T6 median 최소고유값: 투영 전 {min_eig(gm):+.3e} / 투영 후 {min_eig(gp):+.3e}")
    print("   (median은 원소별 연산이라 PSD가 보장되지 않는다 — 주 결과로 쓰지 말 것)")

    print("-" * 62)
    if fails:
        print(f"실패한 점검: {fails}")
        return 1
    print("모든 점검 통과. 다음 단계는 실제 모델에서의 배선 검증이다:")
    print("  bash jobs_robust_check.sh    (mean·K=4 가 기준선 GPTQ와 같은 가중치를 내는지)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
