#!/usr/bin/env python3
"""
compare_ckpt.py — 두 양자화 체크포인트의 가중치를 직접 비교한다.

*** 향후 연구용 코드다. 본 논문의 결과에는 사용되지 않았다. ***

용도: 강건 집계(gptq_robust)를 agg=mean으로 돌리면 기존 GPTQ와 **수학적으로
동일**해야 한다. 그 동일성을 평가 점수로 확인하면 비싸고 둔하므로(점수가
우연히 같을 수도, 미세한 차이가 점수에 묻힐 수도 있다), 저장된 양자화 가중치와
스케일을 텐서 단위로 비교한다.

판정 기준
  - weight_packed / weight (정수 코드) : **완전 일치**를 요구한다.
    GPTQ의 반올림 결정은 이산적이라, Hessian이 같으면 같은 코드가 나와야 한다.
    부동소수점 누적 순서 차이로 극소수 원소가 경계에서 갈릴 수 있으므로,
    불일치 비율이 --code-tol(기본 1e-4 = 0.01%) 이하면 통과로 본다.
  - weight_scale / weight_zero_point / weight_g_idx : 상대오차 --tol 이하.

불일치가 이 한계를 넘으면, "집계 방식 때문"이 아닌 다른 요인이 섞여 있다는 뜻이다.

사용법:
  python compare_ckpt.py <기준선_디렉터리> <비교_디렉터리> [--report out.json]
"""

import argparse
import json
import sys
from pathlib import Path


def load_tensors(d: Path):
    """safetensors 샤드 전체를 {이름: 텐서} 로 읽는다(느리게 순회)."""
    from safetensors import safe_open

    files = sorted(d.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"safetensors 파일이 없다 → {d}")
    out = {}
    for f in files:
        with safe_open(str(f), framework="pt") as fh:
            for k in fh.keys():
                out[k] = fh.get_tensor(k)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("ref", type=Path, help="기준선 체크포인트 디렉터리 (gptq_4bit...)")
    ap.add_argument("cmp", type=Path, help="비교 체크포인트 디렉터리 (gptq_robust_4bit..._mean)")
    ap.add_argument("--tol", type=float, default=1e-5,
                    help="스케일·영점의 허용 상대오차 (기본 1e-5)")
    ap.add_argument("--code-tol", type=float, default=1e-4,
                    help="정수 코드의 허용 불일치 비율 (기본 1e-4)")
    ap.add_argument("--report", type=Path, default=None, help="결과 JSON 저장 경로")
    a = ap.parse_args()

    import torch

    A = load_tensors(a.ref)
    B = load_tensors(a.cmp)

    only_a = sorted(set(A) - set(B))
    only_b = sorted(set(B) - set(A))
    shared = sorted(set(A) & set(B))

    rows, worst_code, worst_rel = [], 0.0, 0.0
    for k in shared:
        ta, tb = A[k], B[k]
        if ta.shape != tb.shape:
            rows.append({"name": k, "verdict": "shape_mismatch",
                         "ref_shape": list(ta.shape), "cmp_shape": list(tb.shape)})
            worst_code = 1.0
            continue

        is_code = ("weight_packed" in k) or (k.endswith(".weight")
                                             and ta.dtype in (torch.int8, torch.int32,
                                                              torch.uint8))
        if is_code:
            mism = float((ta != tb).sum()) / max(ta.numel(), 1)
            worst_code = max(worst_code, mism)
            rows.append({"name": k, "kind": "code", "mismatch_frac": mism})
        else:
            fa, fb = ta.to(torch.float64), tb.to(torch.float64)
            denom = max(float(fa.abs().max()), 1e-30)
            rel = float((fa - fb).abs().max()) / denom
            worst_rel = max(worst_rel, rel)
            rows.append({"name": k, "kind": "float", "max_rel_err": rel})

    ok = (not only_a) and (not only_b) and (worst_code <= a.code_tol) \
        and (worst_rel <= a.tol)

    print(f"기준선: {a.ref}")
    print(f"비교  : {a.cmp}")
    print(f"공통 텐서 {len(shared)}개 | 기준선 전용 {len(only_a)} | 비교 전용 {len(only_b)}")
    print(f"정수 코드 최대 불일치 비율 : {worst_code:.3e}  (허용 {a.code_tol:.1e})")
    print(f"실수 파라미터 최대 상대오차 : {worst_rel:.3e}  (허용 {a.tol:.1e})")
    if only_a or only_b:
        print(f"  누락 예시 — 기준선만: {only_a[:5]} / 비교만: {only_b[:5]}")

    # 가장 어긋난 상위 5개를 남겨, 특정 층에 몰렸는지 흩어졌는지 보게 한다.
    bad = sorted((r for r in rows if r.get("kind") == "code"),
                 key=lambda r: -r.get("mismatch_frac", 0))[:5]
    for r in bad:
        if r.get("mismatch_frac", 0) > 0:
            print(f"  {r['name']}: 불일치 {r['mismatch_frac']:.3e}")

    print("판정:", "일치 (배선 검증 통과)" if ok else "불일치 (원인 조사 필요)")

    if a.report:
        a.report.parent.mkdir(parents=True, exist_ok=True)
        a.report.write_text(json.dumps({
            "ref": str(a.ref), "cmp": str(a.cmp), "passed": ok,
            "worst_code_mismatch_frac": worst_code, "worst_float_rel_err": worst_rel,
            "code_tol": a.code_tol, "tol": a.tol,
            "only_in_ref": only_a, "only_in_cmp": only_b,
            "tensors": rows,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"보고서 저장 → {a.report}")

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
