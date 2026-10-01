#!/usr/bin/env python3
"""
robust_gptq.py — 캘리브레이션 표본 변동을 줄이기 위한 강건 Hessian 집계 GPTQ

*** 향후 연구용 코드다. 본 논문(EXAONE 3.5 7.8B/32B)의 결과에는 사용되지 않았다. ***

--------------------------------------------------------------------------
1. 왜 이 방향인가
--------------------------------------------------------------------------
본 논문 4.3절에서 캘리브레이션 표본 수를 128 → 1024로 늘려도 언어 비대칭
지표 X의 표준편차가 줄지 않는다는 것을 관측했다(1.07 / 0.99 / 1.01 / 1.03,
1/sqrt(n) 이면 0.35배로 줄어야 한다). 즉 변동의 원인이 "표본이 적어서"가
아니라는 뜻이고, 그렇다면 남은 조작 가능한 지점은 **표본을 몇 개 쓰는가**가
아니라 **표본에서 얻은 통계량을 어떻게 합치는가**다.

GPTQ는 캘리브레이션 입력 x로부터 H = 2 * sum_i x_i x_i^T 를 쌓고, 압축 시점에
H / n (즉 표본평균)을 쓴다. 표본평균은 소수의 이상 표본(특이하게 큰 활성화를
유발하는 문장)에 선형적으로 끌려간다. 한국어 instruction 데이터처럼 문장
길이·도메인 편차가 큰 코퍼스에서는 이 민감성이 곧 시드 간 변동으로 나타날 수
있다 — 이것이 본 코드가 검증하려는 가설이다.

가설(H-mit): 캘리브레이션을 K개 부분집합으로 나눠 부분집합별 Hessian을 구한
뒤 표본평균 대신 강건 추정량(기하중앙값 등)으로 합치면, **총 캘리브레이션
예산을 그대로 유지한 채** SD(X)가 줄어든다.

이 가설은 반증 가능하다. 줄지 않으면 "변동의 원인은 집계 방식이 아니다"라는
결과가 되고, 그것 자체가 4.3절의 관측을 보강한다.

--------------------------------------------------------------------------
2. 집계 방식과 PSD 보존
--------------------------------------------------------------------------
부분집합 k의 평균 정규화 Hessian을 M_k = H_k / n_k 라 한다. 모든 M_k는
대칭 준양정부호(PSD)다. GPTQ의 Cholesky 역행렬 계산이 성립하려면 집계
결과도 PSD여야 하는데, 집계 방식에 따라 이 성질이 보존되는지가 다르다.

  mean       : sum_k H_k / sum_k n_k
               -> 기존 GPTQ와 **수학적으로 동일**. 대조군이자 배선 검증용이다.
                  이 설정이 기준선을 재현하지 못하면 이 코드에 버그가 있다는
                  뜻이므로, 어떤 실험보다 먼저 확인해야 한다(robust_selftest.py).
  geomedian  : Weiszfeld 반복으로 구한 기하중앙값.
               가중치가 비음수이고 합이 1인 **볼록결합**이므로 PSD가 구조적으로
               보존된다. 연합학습의 강건 집계(RFA, Pillutla et al.)와 같은 형태다.
               -> 권장 기본값.
  trimmed    : Frobenius 거리 기준으로 양 끝 trim개 부분집합을 버린 뒤 평균.
               남은 것들의 균등 볼록결합이므로 PSD 보존.
  median     : 원소별 중앙값. 구현은 가장 단순하지만 **PSD가 보장되지 않는다**
               (서로 다른 원소가 서로 다른 부분집합에서 오기 때문이다).
               psd_project=True로 고유값 클리핑을 붙일 수 있으나 큰 층에서
               고유분해 비용이 크다. 비교 대상으로만 두고 주 결과로는 쓰지 말 것.

geomedian / trimmed / mean 세 가지는 모두 볼록결합이라 PSD·대칭이 보존되며,
GPTQ의 dampening(percdamp)이 그 위에 추가로 걸린다.

--------------------------------------------------------------------------
3. llmcompressor==0.10.0.3 연결 지점 (실제 소스 확인 결과)
--------------------------------------------------------------------------
- GPTQModifier.calibrate_module(module, args, _output)가 forward 훅으로
  붙어 배치마다 호출되고, self._hessians[module] / self._num_samples[module]에
  누적한다.
- 압축은 compress_module_list()에서
      quantize_weight(..., hessian=self._hessians.pop(m) / self._num_samples.pop(m))
  로 수행된다.
- SequentialPipeline / BasicPipeline 모두 forward 직전에
      session.state.current_batch_idx = batch_idx
  를 설정한다(AWQModifier도 같은 값을 읽어 쓴다). 따라서 훅 안에서 현재 배치가
  몇 번째인지 알 수 있고, 이를 부분집합 번호로 환산할 수 있다.

이 구조를 이용해 **부모의 두 딕셔너리를 그대로 살려 둔다**:
  - self._hessians[module]      : 현재 활성 부분집합의 GPU 누적 버퍼
  - self._subset_H[module][k]   : 부분집합별 누적 결과(기본 CPU 상주)
  - 압축 직전 집계 결과를 self._hessians[module]에 넣고
    self._num_samples[module] = 1.0 로 두면, 부모의 `H / n` 연산이 항등이 되어
    quantize_weight · 로깅 · 오프로딩 · on_finalize 검증을 손대지 않고 재사용한다.

  주의: 그 결과 부모가 찍는 "Quantizing <name> using 1.0 samples" 로그의
  숫자는 무의미하다. 실제 표본 수와 부분집합 구성은 이 클래스가 별도로 찍는다.

- 파이프라인은 DISABLE_QAC_MODIFIERS를 **클래스 이름 문자열**로 비교하므로,
  하위 클래스 이름을 그 목록에 등록하지 않으면 quantization-aware calibration이
  켜진 상태로 돌아 기준선과 조건이 달라진다. 아래에서 import 시점에 등록한다.
  (이 한 줄을 빼면 조용히 다른 실험이 된다 — 가장 위험한 함정이다.)

--------------------------------------------------------------------------
4. 메모리
--------------------------------------------------------------------------
Hessian은 (입력차원)^2 float32다. c_proj의 입력차원은 intermediate_size라
가장 크다.

  7.8B  intermediate 14336 -> 14336^2 * 4B ≈ 0.82 GiB
  32B   intermediate 27392 -> 27392^2 * 4B ≈ 2.80 GiB

K개를 동시에 들고 있으면 이 값의 K배가 필요하다. 그래서 부분집합은 기본적으로
CPU에 두고(subset_device="cpu"), GPU에는 활성 버퍼 하나만 남긴다. 집계 연산도
열 블록 단위로 쪼개(max_chunk_cols) K개를 한꺼번에 스택하지 않는다.

  subset_device="cpu", K=4, 32B 기준 CPU 추가 사용량 ≈ 2.8 GiB × 4 ≈ 11 GiB
  (한 서브그래프의 모듈들이 동시에 살아 있으므로, 층 하나의 모듈 합으로 계산할 것)

--------------------------------------------------------------------------
5. 사용법
--------------------------------------------------------------------------
  # 집계 수식·PSD·강건성 자체 점검 (GPU 불필요, 수 초)
  python robust_selftest.py

  # run_quant_stage.py에 붙이는 방법은 robust_patch.md 참조
  python run_quant_stage.py --stage gptq_robust --model ... \\
      --calib-lang ko --calib-seed 1 --robust-k 4 --robust-agg geomedian
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import torch

__all__ = ["RobustGPTQModifier", "aggregate_hessians", "AGGREGATIONS"]

AGGREGATIONS = ("mean", "median", "trimmed", "geomedian")


# ============================================================
# 집계 연산 (모델·llmcompressor와 독립 — 그래서 단독 검증이 가능하다)
# ============================================================
def _chunks(d: int, max_cols: int):
    for s in range(0, d, max_cols):
        yield s, min(s + max_cols, d)


def _symmetrize(M: torch.Tensor) -> torch.Tensor:
    """수치 오차로 생긴 비대칭을 제거한다. 볼록결합이면 원래 대칭이지만
    누적 과정의 반올림으로 미세하게 깨질 수 있다."""
    return 0.5 * (M + M.t())


def _psd_project(M: torch.Tensor) -> torch.Tensor:
    """고유값을 0에서 클리핑해 PSD로 투영한다. median 집계 전용 보정이며
    d^2 행렬의 고유분해 비용이 크다는 점을 감안해 기본적으로 쓰지 않는다."""
    M = _symmetrize(M)
    evals, evecs = torch.linalg.eigh(M.to(torch.float64))
    evals = torch.clamp(evals, min=0.0)
    out = (evecs * evals.unsqueeze(0)) @ evecs.t()
    return out.to(M.dtype)


def _frob_sq_diff(A: torch.Tensor, B: torch.Tensor, max_cols: int) -> float:
    """||A - B||_F^2 을 열 블록으로 나눠 계산한다(전체 차분 행렬을 만들지 않음)."""
    d = A.shape[1]
    total = 0.0
    for s, e in _chunks(d, max_cols):
        diff = A[:, s:e].to(torch.float32) - B[:, s:e].to(torch.float32)
        total += float(diff.pow(2).sum())
    return total


def _weighted_combine(Ms: List[torch.Tensor], w: List[float],
                      out_device: torch.device, out_dtype: torch.dtype,
                      max_cols: int) -> torch.Tensor:
    """sum_k w_k * M_k. 열 블록 단위로 계산해 K개를 동시에 올리지 않는다."""
    d = Ms[0].shape[1]
    out = torch.zeros((Ms[0].shape[0], d), device=out_device, dtype=out_dtype)
    for s, e in _chunks(d, max_cols):
        acc = torch.zeros((out.shape[0], e - s), device=out_device, dtype=torch.float32)
        for wk, Mk in zip(w, Ms):
            if wk == 0.0:
                continue
            acc += wk * Mk[:, s:e].to(device=out_device, dtype=torch.float32)
        out[:, s:e] = acc.to(out_dtype)
    return out


def aggregate_hessians(
    subset_H: List[torch.Tensor],
    subset_n: List[float],
    aggregation: str = "geomedian",
    trim: int = 1,
    geomed_iters: int = 16,
    geomed_eps: float = 1e-8,
    psd_project: bool = False,
    max_chunk_cols: int = 4096,
    out_device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    부분집합별 누적 Hessian(H_k, 정규화 전)과 표본 수(n_k)로부터
    집계된 평균 Hessian을 만든다. 반환값은 GPTQ가 기대하는 `H / n` 형태다.

    aggregation="mean"은 sum(H_k)/sum(n_k)로 계산하므로, 부분집합 크기가
    불균등해도 기존 GPTQ와 정확히 같은 값이 된다(배선 검증의 근거).
    """
    if aggregation not in AGGREGATIONS:
        raise ValueError(f"알 수 없는 집계 방식: {aggregation} (가능: {AGGREGATIONS})")
    if not subset_H:
        raise ValueError("부분집합 Hessian이 비어 있다.")

    K = len(subset_H)
    dev = out_device if out_device is not None else subset_H[0].device
    dtype = subset_H[0].dtype

    # --- mean: 풀링. 기존 GPTQ와 동일. 부분집합으로 나눈 흔적이 남지 않아야 한다.
    if aggregation == "mean" or K == 1:
        n_tot = float(sum(subset_n))
        if n_tot <= 0:
            raise ValueError("총 표본 수가 0이다.")
        return _weighted_combine(subset_H, [1.0 / n_tot] * K, dev, dtype, max_chunk_cols)

    # 이하 집계는 평균 정규화된 M_k = H_k / n_k 를 대상으로 한다.
    # 원본을 건드리지 않도록 스케일은 사용 시점에 적용한다.
    inv = [1.0 / float(n) if n > 0 else 0.0 for n in subset_n]

    class _Scaled:
        """M_k를 실체화하지 않고 열 블록 슬라이스에만 스케일을 적용하는 얇은 뷰."""
        __slots__ = ("H", "s")

        def __init__(self, H, s):
            self.H, self.s = H, s

        @property
        def shape(self):
            return self.H.shape

        @property
        def device(self):
            return self.H.device

        @property
        def dtype(self):
            return self.H.dtype

        def __getitem__(self, idx):
            return self.H[idx].to(torch.float32) * self.s

        def t(self):
            return self.H.t().to(torch.float32) * self.s

    Ms = [_Scaled(H, s) for H, s in zip(subset_H, inv)]

    # --- median: 원소별 중앙값. PSD 미보장.
    if aggregation == "median":
        d = Ms[0].shape[1]
        out = torch.zeros((Ms[0].shape[0], d), device=dev, dtype=dtype)
        for s, e in _chunks(d, max_chunk_cols):
            stack = torch.stack([M[:, s:e].to(dev) for M in Ms], dim=0)
            out[:, s:e] = torch.median(stack, dim=0).values.to(dtype)
            del stack
        out = _psd_project(out) if psd_project else _symmetrize(out)
        return out

    # --- trimmed / geomedian: 먼저 초기 중심(균등 평균)을 잡는다.
    z = _weighted_combine(Ms, [1.0 / K] * K, dev, dtype, max_chunk_cols)

    if aggregation == "trimmed":
        if trim <= 0 or 2 * trim >= K:
            raise ValueError(f"trim={trim}은 K={K}에 대해 유효하지 않다 (2*trim < K 필요).")
        dists = [math.sqrt(_frob_sq_diff_view(M, z, max_chunk_cols)) for M in Ms]
        order = sorted(range(K), key=lambda i: dists[i])
        keep = order[trim:K - trim]          # 가장 가깝고 가장 먼 것을 trim개씩 버린다
        w = [0.0] * K
        for i in keep:
            w[i] = 1.0 / len(keep)
        out = _weighted_combine(Ms, w, dev, dtype, max_chunk_cols)
        return _symmetrize(out)

    # --- geomedian: Weiszfeld. 볼록결합이므로 PSD·대칭 보존.
    for _ in range(max(1, geomed_iters)):
        dists = [math.sqrt(_frob_sq_diff_view(M, z, max_chunk_cols)) for M in Ms]
        w = [1.0 / max(dk, geomed_eps) for dk in dists]
        tot = sum(w)
        if tot <= 0:
            break
        w = [wk / tot for wk in w]
        z_new = _weighted_combine(Ms, w, dev, dtype, max_chunk_cols)
        shift = math.sqrt(_frob_sq_diff(z_new, z, max_chunk_cols))
        z = z_new
        if shift <= geomed_eps:
            break
    return _symmetrize(z)


def _frob_sq_diff_view(M, B: torch.Tensor, max_cols: int) -> float:
    """_frob_sq_diff의 _Scaled 뷰 대응판."""
    d = B.shape[1]
    total = 0.0
    for s, e in _chunks(d, max_cols):
        diff = M[:, s:e].to(B.device) - B[:, s:e].to(torch.float32)
        total += float(diff.pow(2).sum())
    return total


# ============================================================
# Modifier
# ============================================================
def _make_modifier_class():
    """
    llmcompressor를 import할 수 있는 환경에서만 클래스를 만든다.
    이렇게 분리해 두면 robust_selftest.py가 llmcompressor 없이도 집계
    수식을 검증할 수 있다(이 파일의 상단 절반은 순수 torch만 쓴다).
    """
    from compressed_tensors.offload.dist_utils import is_distributed
    from compressed_tensors.utils import get_execution_device
    from loguru import logger
    from pydantic import PrivateAttr

    from llmcompressor.core import active_session
    from llmcompressor.modifiers.quantization import GPTQModifier
    from llmcompressor.modifiers.quantization.gptq.gptq_quantize import (
        accumulate_hessian,
        make_empty_hessian,
    )
    from llmcompressor.utils.helpers import DISABLE_QAC_MODIFIERS

    class RobustGPTQModifier(GPTQModifier):
        """
        GPTQ의 Hessian 집계를 표본평균에서 강건 추정량으로 교체한다.
        총 캘리브레이션 예산(표본 수)은 기존과 동일하게 유지한다 — 그래야
        "예산을 늘려서 나아진 것"이 아니라 "합치는 방식 때문"이라고 말할 수 있다.

        :param num_subsets: 캘리브레이션을 몇 개로 나눌지(K). K=1이면 기존 GPTQ.
        :param aggregation: mean | median | trimmed | geomedian
        :param subset_assignment: block(연속 블록) | interleave(batch_idx % K).
            데이터셋은 이미 시드로 섞여 있으므로 block으로 충분하고,
            block은 한 번에 활성 부분집합이 하나라 전송 횟수가 K-1로 적다.
        :param subset_device: 부분집합 Hessian 상주 장치. 큰 층에서 GPU가
            모자라므로 기본 cpu.
        :param trim: trimmed 집계에서 양 끝에서 버릴 개수(2*trim < K).
        :param max_chunk_cols: 집계 시 한 번에 처리할 열 수(메모리 상한 조절).
        :param psd_project: median 집계 결과를 PSD로 투영할지. 비용이 크다.
        """

        num_subsets: int = 4
        aggregation: str = "geomedian"
        subset_assignment: str = "block"
        subset_device: str = "cpu"
        trim: int = 1
        geomed_iters: int = 16
        geomed_eps: float = 1e-8
        psd_project: bool = False
        max_chunk_cols: int = 4096
        num_calibration_batches: Optional[int] = None

        _subset_H: Dict[torch.nn.Module, Dict[int, torch.Tensor]] = PrivateAttr(
            default_factory=dict)
        _subset_n: Dict[torch.nn.Module, Dict[int, float]] = PrivateAttr(
            default_factory=dict)
        _active_k: Dict[torch.nn.Module, int] = PrivateAttr(default_factory=dict)
        _n_batches: int = PrivateAttr(default=0)
        _reported: bool = PrivateAttr(default=False)

        # ---------- 생명주기 ----------
        def on_start(self, state, event, **kwargs):
            if self.aggregation not in AGGREGATIONS:
                raise ValueError(
                    f"aggregation={self.aggregation} 은 유효하지 않다 (가능: {AGGREGATIONS})")
            if self.num_subsets < 1:
                raise ValueError("num_subsets는 1 이상이어야 한다.")
            if self.subset_assignment not in ("block", "interleave"):
                raise ValueError("subset_assignment는 block 또는 interleave여야 한다.")
            if self.aggregation == "trimmed" and 2 * self.trim >= self.num_subsets:
                raise ValueError(
                    f"trimmed 집계에는 2*trim < K 가 필요하다 (trim={self.trim}, K={self.num_subsets})")

            # 총 배치 수: 명시값 우선, 없으면 캘리브레이션 dataloader에서 추론.
            n = self.num_calibration_batches
            if not n:
                calib = getattr(getattr(state, "data", None), "calib", None)
                try:
                    n = len(calib)
                except Exception:
                    n = 0
            self._n_batches = int(n or 0)
            if self.subset_assignment == "block" and self._n_batches <= 0:
                raise RuntimeError(
                    "block 배정에는 총 배치 수가 필요하다. dataloader 길이를 알 수 없으면 "
                    "num_calibration_batches를 명시하거나 subset_assignment='interleave'를 쓸 것.")
            if self._n_batches and self._n_batches < self.num_subsets:
                raise ValueError(
                    f"배치 수({self._n_batches})가 부분집합 수({self.num_subsets})보다 적다.")

            logger.info(
                f"[RobustGPTQ] K={self.num_subsets} agg={self.aggregation} "
                f"assign={self.subset_assignment} batches={self._n_batches} "
                f"subset_device={self.subset_device}")
            super().on_start(state, event, **kwargs)

        # ---------- 부분집합 번호 ----------
        def _subset_index(self, batch_idx: int) -> int:
            K = self.num_subsets
            if K == 1:
                return 0
            if batch_idx < 0:
                # 파이프라인이 current_batch_idx를 설정하지 않은 경로.
                # 조용히 한 부분집합에 몰아넣으면 실험이 무효가 되므로 즉시 멈춘다.
                raise RuntimeError(
                    "current_batch_idx를 읽을 수 없다. 이 llmcompressor 파이프라인은 "
                    "부분집합 배정을 지원하지 않는다(sequential/basic 파이프라인 필요).")
            if self.subset_assignment == "interleave":
                return batch_idx % K
            per = max(1, math.ceil(self._n_batches / K))
            return min(batch_idx // per, K - 1)

        # ---------- 활성 버퍼를 부분집합으로 이동(가산 병합) ----------
        def _stash_active(self, module: torch.nn.Module) -> None:
            k = self._active_k.get(module)
            if k is None:
                return
            H = self._hessians.get(module)
            n = self._num_samples.get(module)
            if H is None or n is None:
                return
            n_val = float(n)
            if n_val <= 0:
                return
            tgt = torch.device(self.subset_device)
            store = self._subset_H[module]
            if k in store:
                store[k] += H.to(tgt)
            else:
                store[k] = H.to(tgt).clone()
            self._subset_n[module][k] = self._subset_n[module].get(k, 0.0) + n_val
            # 활성 버퍼 초기화(형상·장치 유지)
            H.zero_()
            n.zero_()

        # ---------- 훅 ----------
        def calibrate_module(self, module, args, _output):
            inp = args[0]
            sess = active_session()
            bidx = getattr(sess.state, "current_batch_idx", -1) if sess else -1
            k = self._subset_index(bidx)

            if module not in self._num_samples:
                init_device = ("cpu" if self.offload_hessians
                               else get_execution_device(module))
                self._hessians[module] = make_empty_hessian(module, device=init_device)
                self._num_samples[module] = torch.zeros(
                    tuple(), device=get_execution_device(module))
                self._subset_H[module] = {}
                self._subset_n[module] = {}
                self._active_k[module] = k

            if self._active_k[module] != k:
                with self._maybe_onload_hessian(module):
                    self._stash_active(module)
                self._active_k[module] = k

            with self._maybe_onload_hessian(module):
                self._hessians[module], self._num_samples[module] = accumulate_hessian(
                    inp, module, self._hessians[module], self._num_samples[module])

        # ---------- 압축 직전 집계 ----------
        def compress_modules(self):
            if is_distributed():
                raise RuntimeError(
                    "RobustGPTQModifier는 분산 실행을 지원하지 않는다. 부분집합 Hessian이 "
                    "랭크별로 나뉘면 집계 의미가 달라지므로 단일 프로세스로 실행할 것.")
            self._materialize()
            super().compress_modules()

        def _materialize(self) -> None:
            for module in list(self._num_samples.keys()):
                with self._maybe_onload_hessian(module):
                    self._stash_active(module)
                self._active_k.pop(module, None)

                store = self._subset_H.pop(module, {})
                counts = self._subset_n.pop(module, {})
                ks = sorted(store.keys())
                if not ks:
                    raise RuntimeError(
                        f"{self._module_names.get(module)}: 부분집합 Hessian이 비어 있다.")

                Hs = [store[k] for k in ks]
                ns = [counts[k] for k in ks]

                agg = self.aggregation
                if len(ks) < self.num_subsets:
                    # 배치가 모자라 일부 부분집합이 비었다. 조용히 다른 실험이 되지
                    # 않도록 경고하고, trimmed는 조건을 만족하지 못하면 중단한다.
                    logger.warning(
                        f"[RobustGPTQ] {self._module_names.get(module)}: "
                        f"부분집합 {len(ks)}/{self.num_subsets}개만 채워졌다.")
                    if agg == "trimmed" and 2 * self.trim >= len(ks):
                        raise RuntimeError("trimmed 집계 조건(2*trim < 실제 부분집합 수) 위반.")

                target = (torch.device("cpu") if self.offload_hessians
                          else get_execution_device(module))
                H_agg = aggregate_hessians(
                    Hs, ns,
                    aggregation=agg,
                    trim=self.trim,
                    geomed_iters=self.geomed_iters,
                    geomed_eps=self.geomed_eps,
                    psd_project=self.psd_project,
                    max_chunk_cols=self.max_chunk_cols,
                    out_device=target,
                )
                del Hs, store

                if not self._reported:
                    logger.info(
                        f"[RobustGPTQ] 집계 예시 {self._module_names.get(module)}: "
                        f"부분집합 표본 수 {[int(x) for x in ns]} "
                        f"(총 {int(sum(ns))}) → agg={agg}")
                    self._reported = True

                # 부모의 `H / n` 를 항등으로 만든다. 이 때문에 부모가 찍는
                # "using 1.0 samples" 로그의 숫자는 무의미하다(위 로그가 실제 값).
                self._hessians[module] = H_agg
                self._num_samples[module] = torch.ones(
                    tuple(), device=self._num_samples[module].device)

        def on_finalize(self, state, **kwargs) -> bool:
            self._subset_H = dict()
            self._subset_n = dict()
            self._active_k = dict()
            return super().on_finalize(state, **kwargs)

    # *** 반드시 필요: 파이프라인은 클래스 이름 문자열로 QAC 비활성화를 판단한다.
    #     등록하지 않으면 quantization-aware calibration이 켜진 채 돌아
    #     기준선 GPTQ와 조건이 달라진다(조용한 교란). ***
    if "RobustGPTQModifier" not in DISABLE_QAC_MODIFIERS:
        DISABLE_QAC_MODIFIERS.append("RobustGPTQModifier")

    return RobustGPTQModifier


try:  # llmcompressor가 있는 환경에서만 클래스를 노출한다
    RobustGPTQModifier = _make_modifier_class()
except Exception as _e:  # noqa: BLE001
    _IMPORT_ERROR = _e

    class RobustGPTQModifier:  # type: ignore[no-redef]
        def __init__(self, *a, **k):
            raise RuntimeError(
                f"llmcompressor를 불러올 수 없어 RobustGPTQModifier를 쓸 수 없다: {_IMPORT_ERROR}")


def build_robust_recipe(k: int = 4, aggregation: str = "geomedian",
                        calib_n: int = 256, **kw):
    """
    run_quant_stage.build_recipe()의 gptq 분기를 대체할 레시피를 만든다.
    W4A16·group 128·ignore=lm_head 등 나머지 조건은 기준선과 동일하게 둔다 —
    비교 가능성이 이 실험의 전부이므로 여기서 다른 것을 바꾸면 안 된다.
    """
    return [RobustGPTQModifier(
        targets="Linear", scheme="W4A16", ignore=["lm_head"],
        num_subsets=k, aggregation=aggregation,
        num_calibration_batches=calib_n,
        **kw,
    )]
