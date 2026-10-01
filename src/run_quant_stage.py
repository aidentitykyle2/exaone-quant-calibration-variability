#!/usr/bin/env python3
"""
EXAONE 3.5 / Solar 양자화 비교 실험 스크립트 - 단일 venv 통합판
- 네 가지 4-bit 양자화 기법(AWQ, GPTQ, QuaRot-style rotation, SpinQuant-style
  rotation)을 모두 llmcompressor 하나로 통일해 구현한다. 개별 라이브러리
  (autoawq, auto-gptq)는 둘 다 유지보수가 종료되어 vLLM과의 버전 호환이
  불안정하므로 더 이상 사용하지 않는다.
- 각 단계(bf16 / awq / gptq / quarot / spinquant)는 독립 프로세스로 실행해
  GPU 메모리 누수·OOM 위험을 제거한다. venv는 하나만 있으면 된다.
- 사용법:
    export CUDA_VISIBLE_DEVICES=0
    python run_quant_stage.py --stage bf16
    python run_quant_stage.py --stage awq
    python run_quant_stage.py --stage gptq
    python run_quant_stage.py --stage quarot
    python run_quant_stage.py --stage spinquant
  또는 동봉된 run_all_stages.sh 로 다섯 단계를 순차 실행 (각각 별도 프로세스,
  venv 전환 없음)
"""

import os
import sys
import gc
import json
import time
import re
import platform
import argparse
import traceback
import tempfile
import numpy as np
from pathlib import Path
from datetime import datetime

import torch

# ============================================================
# 설정
# ============================================================
DEFAULT_MODEL_ID = "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct"
# Solar 실험 시: --model upstage/SOLAR-10.7B-Instruct-v1.0

# EXAONE 3.5-7.8B-Instruct 저장소는 2026-02-06 커밋(553ea250...)에서
# config.json을 Transformers v5의 rope_parameters 방식으로 갱신했다.
# 우리 환경은 llmcompressor==0.10.0.3이 요구하는 transformers<=4.57.6(v4)에
# 묶여 있으므로, 그 이전 리비전(0ff6b5ec7c13b049b253a16a889aa269e6b79a94,
# 2024-12-11)으로 고정해야 한다. 모델을 바꾸면 이 기본값도 그에 맞게
# --revision으로 재확인해야 한다.
DEFAULT_REVISION = "0ff6b5ec7c13b049b253a16a889aa269e6b79a94"

BATCH_SIZE = 8
MAX_NEW_TOKENS = 64
NUM_WARMUP = 1
NUM_RUNS = 3

# 캘리브레이션 데이터셋 정의. 언어별 ablation을 위해 두 언어를 지원한다.
#   ko: KoAlpaca (한국어 instruction 데이터)
#   en: C4 영어 부분집합 — 공개된 4비트 체크포인트 대부분이 영어 웹 텍스트로
#       캘리브레이션되어 있어, 실무에서 흔히 쓰이는 조건을 재현한다.
CALIB_DATASETS = {
    "ko": {
        "path": "beomi/KoAlpaca-v1.1a",
        "name": None,
        "split": "train",
        "text_fields": ["instruction", "output"],  # 두 필드를 이어붙여 사용
    },
    "en": {
        "path": "allenai/c4",
        "name": "en",
        "split": "train",
        "text_fields": ["text"],
        "streaming": True,  # C4 전체는 매우 크므로 스트리밍으로 앞부분만 사용
    },
    # 언어/도메인 교란 분리용 대조군.
    # ko(KoAlpaca)는 한국어 instruction, en(C4)은 영어 웹 텍스트라 두 조건은
    # 언어와 도메인이 동시에 다르다. en_instruct(Alpaca)를 추가하면
    #   ko vs en_instruct  -> 도메인 고정, 언어만 다름 (순수 언어 효과)
    #   en_instruct vs en  -> 언어 고정, 도메인만 다름 (순수 도메인 효과)
    # 두 대조를 모두 얻을 수 있다. 필드 처리도 KoAlpaca와 동일하게
    # instruction + output을 이어붙여 형식 차이를 최소화한다.
    "en_instruct": {
        "path": "tatsu-lab/alpaca",
        "name": None,
        "split": "train",
        "text_fields": ["instruction", "output"],
    },
    # 혼합 조건: 두 언어를 절반씩 섞어, 단일 언어 캘리브레이션 대비
    # 양쪽 언어를 동시에 보존할 수 있는지 확인한다.
    "mixed": {"mix": ["ko", "en"]},
}
DEFAULT_CALIB_LANG = "ko"
CALIB_DATASET = CALIB_DATASETS["ko"]["path"]  # 하위 호환용(환경 기록에만 사용)
NUM_CALIB_SAMPLES = 256
MAX_SEQ_LENGTH = 2048

# 평가 태스크. kmmlu 외 항목은 lm-evaluation-harness 버전에 따라 정확한
# 태스크명이 다를 수 있으므로, run_eval.sh 실행 전 `lm_eval --tasks list`로
# 존재 여부를 확인할 것을 권장한다.
EVAL_TASKS = ",".join([
    "kmmlu",
    "kobest_boolq",
    "kobest_copa",
    "kobest_wic",
    "kobest_hellaswag",
    "kobest_sentineg",
    "haerae",  # 태스크명 확인 필요
])

STAGES = ["bf16", "awq", "gptq", "quarot", "spinquant"]
METHOD_NAMES = {
    "bf16": "BF16",
    "awq": "AWQ-4bit",
    "gptq": "GPTQ-4bit",
    "quarot": "QuaRot-4bit",
    "spinquant": "SpinQuant-4bit",
}


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


# ============================================================
# 실행 전 안전장치
# ============================================================
def enforce_cuda_visible_devices() -> None:
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        if os.environ.get("ALLOW_UNSET_CUDA_VISIBLE_DEVICES") == "1":
            log("경고: CUDA_VISIBLE_DEVICES 미설정 상태로 강제 진행합니다.")
            return
        log("오류: CUDA_VISIBLE_DEVICES가 설정되어 있지 않습니다.")
        log("  예: export CUDA_VISIBLE_DEVICES=0")
        sys.exit(1)


def check_gpu(min_free_ratio: float = 0.3) -> bool:
    if not torch.cuda.is_available():
        log("CUDA를 사용할 수 없습니다.")
        return False
    name = torch.cuda.get_device_name(0)
    free, total = torch.cuda.mem_get_info(0)
    free_gb, total_gb = free / 1024**3, total / 1024**3
    log(f"GPU 0: {name} | 가용: {free_gb:.1f} GB / 전체: {total_gb:.1f} GB")
    if free / total < min_free_ratio:
        log(f"경고: 가용 메모리 비율이 {min_free_ratio:.0%} 미만입니다.")
        return False
    return True


def _nvml_device_index() -> int:
    """
    CUDA_VISIBLE_DEVICES의 첫 번째 항목을 실제 물리 GPU 인덱스로 사용한다.
    NVML은 CUDA_VISIBLE_DEVICES를 자동으로 반영하지 않으므로 직접 파싱해야 한다.
    """
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    first = cvd.split(",")[0].strip()
    return int(first) if first else 0


def get_gpu_memory_used_gb() -> float:
    """
    torch.cuda.memory_allocated()는 '현재 파이썬 프로세스'가 잡은 CUDA 메모리만
    본다. vLLM V1 엔진은 실제 모델을 별도 서브프로세스(EngineCore)에서 로드·
    실행하므로, 메인 프로세스 기준 torch.cuda 측정치는 항상 0에 가깝게 나와
    실제 점유량을 반영하지 못한다. pynvml로 GPU 자체를 프로세스 외부에서
    조회하면 어느 프로세스가 메모리를 잡고 있든 정확한 값을 얻는다.
    """
    import pynvml
    pynvml.nvmlInit()
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(_nvml_device_index())
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return info.used / 1024**3
    finally:
        pynvml.nvmlShutdown()


# ============================================================
# 결과 파일 (단계별 증분 저장 — 읽고 병합 후 다시 쓰기)
# ============================================================
def load_or_init_results(results_file: Path) -> dict:
    if results_file.exists():
        with open(results_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"environment": {}, "efficiency": [], "errors": []}


def save_results(results_file: Path, data: dict) -> None:
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    log(f"결과 저장 → {results_file}")


def get_environment_info(model_id: str) -> dict:
    info = {
        "timestamp": datetime.now().isoformat(),
        "model_id": model_id,
        "pytorch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "(unset)"),
    }
    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_memory_total_gb"] = round(
            torch.cuda.get_device_properties(0).total_memory / 1024**3, 2
        )
        info["cuda_device_count"] = torch.cuda.device_count()

    for pkg_name, attr in [("transformers", "transformers"),
                            ("vllm", "vllm"),
                            ("llmcompressor", "llmcompressor"),
                            ("compressed_tensors", "compressed_tensors")]:
        try:
            mod = __import__(pkg_name)
            info[attr] = getattr(mod, "__version__", "unknown")
        except ImportError:
            info[attr] = None
    return info


# ============================================================
# 데이터
# ============================================================
def load_calib_dataset(lang: str = DEFAULT_CALIB_LANG,
                        n_samples: int = NUM_CALIB_SAMPLES,
                        seed: int = 0,
                        manifest: list = None,
                        provenance: dict = None):
    """
    llmcompressor.oneshot()이 기대하는 HF Dataset("text" 컬럼)로 반환한다.

    seed == 0 : 데이터셋 앞에서부터 결정적으로 추출한다. 초기 실험(시드 도입 전)과
                완전히 동일한 샘플을 재현하므로 기존 결과를 seed 0으로 재사용할 수 있다.
    seed  > 0 : 데이터셋을 해당 시드로 섞은 뒤 추출한다. 캘리브레이션 표본 추출에
                따른 변동을 추정하기 위한 반복 실험용이다.

    lang으로 캘리브레이션 언어를 선택한다. 언어별 ablation의 핵심 축이다 —
    한국어 캘리브레이션과 영어 캘리브레이션이 언어별 성능 보존에 미치는
    영향을 비교하기 위해, 데이터 출처 외의 모든 조건(샘플 수, 최소 길이,
    최대 시퀀스 길이)은 동일하게 유지한다.

    manifest가 주어지면 실제로 뽑힌 표본을 원본 식별자와 함께 순서대로 기록한다.
    provenance가 주어지면 원본 데이터셋의 출처(경로, 캐시 파일 = 데이터셋 리비전)를 기록한다.
    두 인자는 선택된 표본에 영향을 주지 않는다(셔플은 길이와 시드로만 결정된다).
    """
    from datasets import load_dataset

    if lang not in CALIB_DATASETS:
        raise ValueError(f"알 수 없는 캘리브레이션 언어: {lang} "
                         f"(가능: {list(CALIB_DATASETS.keys())})")
    cfg = CALIB_DATASETS[lang]
    log(f"캘리브레이션 데이터 로딩 (lang={lang}, n={n_samples}, seed={seed})...")

    # 혼합 조건: 구성 언어들에서 균등하게 나눠 담는다. 총 샘플 수는 단일 언어
    # 조건과 동일하게 유지해야 "샘플 수"가 교란변수가 되지 않는다.
    if "mix" in cfg:
        from datasets import Dataset
        langs = cfg["mix"]
        per_lang = n_samples // len(langs)
        remainder = n_samples - per_lang * len(langs)
        texts = []
        for i, sub_lang in enumerate(langs):
            take = per_lang + (1 if i < remainder else 0)
            sub = load_calib_dataset(sub_lang, take, seed=seed,
                                     manifest=manifest, provenance=provenance)
            texts.extend(sub["text"])
        ds = Dataset.from_dict({"text": texts})
        log(f"캘리브레이션 샘플 수: {len(ds)} (혼합: {langs})")
        return ds

    fields = cfg["text_fields"]

    def _to_text(ex):
        parts = [str(ex.get(f, "") or "") for f in fields]
        return {"text": " ".join(parts).strip()}

    if cfg.get("streaming"):
        # 스트리밍 데이터셋은 select()를 쓸 수 없어, 필요한 만큼만 순회해 모은다.
        stream = load_dataset(cfg["path"], cfg["name"], split=cfg["split"],
                               streaming=True)
        if seed > 0:
            # 스트리밍은 전체 셔플이 불가능하므로 버퍼 셔플을 쓴다.
            stream = stream.shuffle(seed=seed, buffer_size=10_000)
        texts = []
        for pos, ex in enumerate(stream):
            t = _to_text(ex)["text"]
            if len(t) > 30:
                texts.append(t)
                if manifest is not None:
                    manifest.append({"lang": lang, "source": cfg["path"],
                                     "source_id": ex.get("url") or f"stream_pos:{pos}", "text": t})
            if len(texts) >= n_samples:
                break
        if provenance is not None:
            provenance[lang] = {"path": cfg["path"], "name": cfg["name"], "split": cfg["split"],
                                "streaming": True, "shuffle_buffer": 10_000 if seed > 0 else None}
        from datasets import Dataset
        ds = Dataset.from_dict({"text": texts})
    else:
        ds = load_dataset(cfg["path"], cfg["name"], split=cfg["split"])
        if provenance is not None:
            cache = ds.cache_files[0]["filename"] if ds.cache_files else None
            provenance[lang] = {"path": cfg["path"], "name": cfg["name"], "split": cfg["split"],
                                "num_rows": len(ds), "cache_file": cache}
        # 원본 행 번호를 열로 붙여 셔플·선택·필터 뒤에도 추적한다.
        ds = ds.add_column("__row", list(range(len(ds))))
        if seed > 0:
            ds = ds.shuffle(seed=seed)
        ds = ds.select(range(min(n_samples * 2, len(ds))))  # 필터링 여유분 확보
        ds = ds.map(_to_text, remove_columns=[c for c in ds.column_names if c != "__row"])
        ds = ds.filter(lambda ex: len(ex["text"]) > 30)
        if len(ds) > n_samples:
            ds = ds.select(range(n_samples))
        if manifest is not None:
            for row, t in zip(ds["__row"], ds["text"]):
                manifest.append({"lang": lang, "source": cfg["path"], "source_id": int(row), "text": t})
        ds = ds.remove_columns("__row")   # oneshot()에는 text 열만 넘긴다

    log(f"캘리브레이션 샘플 수: {len(ds)}")
    return ds


def calib_set_name(lang: str, n: int, seed: int) -> str:
    """캘리브레이션 세트 식별 이름. 표본은 (언어, 크기, 시드)로만 결정되므로 기법과 무관하다."""
    return f"calib_{lang}_n{n}_s{seed}"


def save_calib_manifest(out_root: Path, lang: str, n: int, seed: int,
                        manifest: list, provenance: dict, include_text: bool = True,
                        origin: str = "runtime") -> Path:
    """
    실제 사용한 캘리브레이션 표본을 저장한다(재현성 확보용).
      <이름>.jsonl      표본별 순번, 언어, 원본 데이터셋, 원본 식별자, 본문 해시, (선택) 본문
      <이름>.meta.json  조건, 표본 수, 세트 전체 해시, 추출 규칙, 원본 출처, 라이브러리 버전

    시드만으로는 라이브러리·플랫폼 버전에 따라 같은 표본이 재현되지 않을 수 있으므로
    (Williams & Aletras, 2024), 표본 자체와 식별자를 함께 남긴다.
    include_text=False이면 본문 없이 식별자와 해시만 저장한다(원본 라이선스상 재배포가
    제한될 때 사용).
    """
    import hashlib
    import datasets as _ds

    out_root.mkdir(parents=True, exist_ok=True)
    name = calib_set_name(lang, n, seed)
    texts = [m["text"] for m in manifest]
    with open(out_root / f"{name}.jsonl", "w", encoding="utf-8") as f:
        for i, m in enumerate(manifest):
            rec = {"i": i, "lang": m["lang"], "source": m["source"], "source_id": m["source_id"],
                   "sha1": hashlib.sha1(m["text"].encode("utf-8")).hexdigest(),
                   "chars": len(m["text"])}
            if include_text:
                rec["text"] = m["text"]
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    meta = {
        "calib_lang": lang, "calib_n": n, "calib_seed": seed, "count": len(manifest),
        "set_sha256": hashlib.sha256("\x1e".join(texts).encode("utf-8")).hexdigest(),
        "selection_rule": ("seed 0: 원본 순서 앞에서부터 / seed>0: datasets.shuffle(seed) 후 앞에서부터. "
                           "n×2개를 먼저 고른 뒤 30자 초과 표본만 남기고 n개로 자른다. "
                           "혼합 조건은 구성 언어별로 n을 균등 분할한다."),
        "sources": provenance,
        "datasets_version": _ds.__version__,
        "text_included": include_text,
        # runtime       : 양자화 실행 중에 기록
        # reconstructed : 기록 기능 도입 전 실행을 같은 규칙으로 다시 추출해 재구성
        "origin": origin,
    }
    (out_root / f"{name}.meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"캘리브레이션 표본 기록 → {out_root / name}.jsonl ({len(manifest)}개, "
        f"sha256 {meta['set_sha256'][:12]}...)")
    return out_root / f"{name}.jsonl"


# 하위 호환용 별칭 (기존 호출부가 남아 있을 경우 대비)
def load_korean_calib_dataset(n_samples: int = NUM_CALIB_SAMPLES):
    return load_calib_dataset("ko", n_samples)


def get_prompts(batch_size: int = BATCH_SIZE) -> list:
    base = [
        "대한민국의 수도는?",
        "인공지능이 사회에 미치는 영향을 간략히 설명해 주세요.",
        "한국의 지방자치제도와 중앙정부의 관계를 구체적인 사례와 함께 서술해 주세요.",
    ]
    return (base * ((batch_size // len(base)) + 1))[:batch_size]


# ============================================================
# 양자화 — AWQ / GPTQ / QuaRot / SpinQuant을 llmcompressor 하나로 통일
# ============================================================
def _exaone_awq_mappings():
    """
    EXAONE(GPT-2/NeoX 계열 명명 규칙)은 llmcompressor의 기본 AWQ 매핑이
    가정하는 Llama식 이름(input_layernorm, self_attn.*, gate_proj/up_proj)과
    맞지 않아 자동 추론이 실패한다. 실제 모듈 트리
    (transformer.h.N.ln_1 / attn.attention.{q,k,v}_proj / attn.attention.out_proj /
    ln_2 / mlp.c_fc_0 / mlp.c_fc_1 / mlp.c_proj)와 forward 코드
    (output = c_proj(act(c_fc_0(x)) * c_fc_1(x)))를 직접 확인해 수동으로 지정한다.
    c_fc_0은 활성화를 거치므로(gate_proj에 해당) 두 번째 매핑에서 제외하고,
    활성화 없이 선형적으로 전달되는 c_fc_1(up_proj에 해당)만 c_proj(down_proj)로
    매핑해야 AWQ의 스케일 상쇄가 수학적으로 성립한다.

    v_proj -> out_proj 매핑은 의도적으로 제외한다. EXAONE 3.5는 GQA
    (num_key_value_heads < num_attention_heads)를 사용해 v_proj 출력 차원이
    out_proj 입력 차원보다 작고, 그 사이에 repeat_kv라는 비-가중치 확장
    연산이 끼어 있어 단순 스무딩 나눗셈이 차원 불일치로 깨진다
    ("size of tensor a (1024) must match tensor b (4096)" 오류로 확인됨).
    이 매핑은 네 매핑 중 부가적인 것이고(핵심은 ln_2 -> gate/up), GQA 확장
    관계를 정확히 반영하는 커스텀 로직 없이는 안전하게 처리할 수 없으므로
    생략한다.
    """
    from llmcompressor.modifiers.awq import AWQMapping
    return [
        AWQMapping(
            "re:.*ln_1$",
            ["re:.*attn.attention.q_proj$", "re:.*attn.attention.k_proj$",
             "re:.*attn.attention.v_proj$"],
        ),
        AWQMapping("re:.*ln_2$", ["re:.*mlp.c_fc_0$", "re:.*mlp.c_fc_1$"]),
        AWQMapping("re:.*mlp.c_fc_1$", ["re:.*mlp.c_proj$"]),
    ]


def _exaone_spinquant_mappings():
    """
    SpinQuantModifier도 AWQ와 마찬가지로 SPINQUANT_MAPPING_REGISTRY /
    NORM_MAPPING_REGISTRY에 "LlamaForCausalLM"만 등록되어 있어, EXAONE에서는
    자동 추론이 Llama식 이름(self_attn, input_layernorm 등)으로 실패한다.
    llmcompressor==0.10.0.3 소스(SpinQuantMapping, NormMapping 정의)를 직접
    확인해 EXAONE 실제 구조에 맞게 두 매핑을 수동 구성한다.

    R2(어텐션 v/o 회전)는 head_dim(=hidden_size/num_attention_heads=128) 단위
    블록 대각 행렬로 헤드별 독립 적용되므로, AWQ 때와 달리 GQA로 인한
    v_proj/o_proj 차원 불일치 문제는 여기서는 발생하지 않는다.

    mlp_in/mlp_out은 AWQ와 달리 활성화 비선형 여부를 구분할 필요가 없다 —
    회전은 스케일 상쇄 트릭이 아니라 residual stream 경계에 일관되게 적용되는
    직교 변환이므로, c_fc_0(gate)과 c_fc_1(up) 둘 다 mlp_in에 포함된다.

    attn 필드(R3의 q_attn/k_cache 훅 대상)는 실제 어텐션 연산(rotary 포함)이
    일어나는 transformer.h.N.attn.attention으로 지정했다 — 이 지점은 다른
    필드들처럼 named_modules() 트리로 직접 확인하지 못했으므로, R3 단계에서
    오류가 나면 가장 먼저 의심하고 transformer.h.N.attn(바깥쪽 wrapper)으로
    바꿔 재시도할 지점이다.
    """
    from llmcompressor.modifiers.transform.spinquant.mappings import SpinQuantMapping
    from llmcompressor.modifiers.transform.spinquant.norm_mappings import NormMapping

    mapping = SpinQuantMapping(
        embedding="re:.*wte$",
        attn="re:.*attn.attention$",
        attn_q="re:.*attn.attention.q_proj$",
        attn_k="re:.*attn.attention.k_proj$",
        attn_v="re:.*attn.attention.v_proj$",
        attn_o="re:.*attn.attention.out_proj$",
        mlp_in=["re:.*mlp.c_fc_0$", "re:.*mlp.c_fc_1$"],
        mlp_out="re:.*mlp.c_proj$",
        lm_head="lm_head",
    )
    norm_mappings = [
        NormMapping(
            norm="re:.*ln_1$",
            linears=["re:.*attn.attention.q_proj$", "re:.*attn.attention.k_proj$",
                     "re:.*attn.attention.v_proj$"],
        ),
        NormMapping(
            norm="re:.*ln_2$",
            linears=["re:.*mlp.c_fc_0$", "re:.*mlp.c_fc_1$"],
        ),
        NormMapping(
            norm="transformer.ln_f",
            linears=["lm_head"],
        ),
    ]
    return mapping, norm_mappings


def _build_exaone_transform_config(mapping, head_dim: int, rotations,
                                    transform_type: str, precision):
    """
    SpinQuantModifier.on_initialize()가 transform_config가 None일 때
    self.mappings/self.norm_mappings를 무조건 자동 추론값으로 덮어써버려서
    (llmcompressor==0.10.0.3의 실제 동작 — mappings 생성자 인자를 그냥
    무시한다), 매핑을 전달하는 것만으로는 EXAONE 지원이 되지 않았다.
    _create_r1_scheme~_create_r4_scheme(base.py)의 로직을 그대로 복제하되
    self.mappings 대신 우리가 만든 EXAONE mapping을 사용해 transform_config를
    직접 구성한다. transform_config를 명시적으로 넘기면 on_initialize가
    조기 반환하며 self.mappings/self.norm_mappings를 건드리지 않으므로,
    이후 on_start()의 _center_embeddings/_fuse_norms도 우리 매핑 그대로 쓴다.
    """
    from compressed_tensors.transform import TransformArgs, TransformConfig, TransformScheme

    config_groups = {}

    if "R1" in rotations:
        config_groups["R1"] = TransformScheme(
            type=transform_type, randomize=False, requires_grad=False,
            precision=precision, head_dim=None,
            apply=[
                TransformArgs(
                    targets=[mapping.embedding, mapping.attn_o, *mapping.mlp_out],
                    location="weight_output",
                ),
                TransformArgs(
                    targets=[mapping.attn_q, mapping.attn_k, mapping.attn_v,
                             *mapping.mlp_in, mapping.lm_head],
                    location="weight_input", inverse=True,
                ),
            ],
        )
    if "R2" in rotations:
        config_groups["R2"] = TransformScheme(
            type=transform_type, randomize=False, requires_grad=False,
            precision=precision, head_dim=head_dim,
            apply=[
                TransformArgs(targets=[mapping.attn_v], location="weight_output"),
                TransformArgs(targets=[mapping.attn_o], location="weight_input", inverse=True),
            ],
        )
    if "R3" in rotations:
        config_groups["R3"] = TransformScheme(
            type=transform_type, randomize=False, requires_grad=False,
            precision=precision, head_dim=head_dim,
            apply=[
                TransformArgs(targets=[mapping.attn], location="q_attn"),
                TransformArgs(targets=[mapping.attn], location="k_cache"),
            ],
        )
    if "R4" in rotations:
        config_groups["R4"] = TransformScheme(
            type=transform_type, randomize=False, requires_grad=False,
            precision=precision, head_dim=None,
            apply=[
                TransformArgs(targets=[*mapping.mlp_out], location="input"),
                TransformArgs(targets=[*mapping.mlp_out], location="weight_input", inverse=True),
            ],
        )
    return TransformConfig(config_groups=config_groups)


def _exaone_head_dim(model_id: str, revision: str) -> int:
    """compressed_tensors.utils.get_head_dim과 동일한 규칙으로 계산한다."""
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(model_id, revision=revision, trust_remote_code=True)
    if hasattr(config, "head_dim") and config.head_dim is not None:
        return config.head_dim
    return config.hidden_size // config.num_attention_heads


def build_recipe(method: str, model_id: str = None, revision: str = None):
    """
    method별 llmcompressor recipe를 구성한다.
    - awq       : AWQModifier (Activation-aware, W4A16)
    - gptq      : GPTQModifier (W4A16)
    - quarot    : SpinQuantModifier(learnable=False) + GPTQModifier
                  (QuaRot 원 논문 구성 = 회전 + GPTQ)
                  — 학습 없는 고정 Hadamard 회전 (QuaRot 원 논문과 동일한 방식)
    - spinquant : SpinQuantModifier(learnable=True) + QuantizationModifier
                  — 캘리브레이션 데이터로 학습된 회전 (Cayley 최적화)
                  ** llmcompressor==0.10.0.3에는 learnable=True가 아직
                  구현되어 있지 않다(NotImplementedError). 이 버전에서는
                  spinquant 단계를 실행할 수 없으며, 별도로 라이브러리
                  버전을 재검토하기 전까지는 quarot만 유효하다. **
    네 기법 모두 최종적으로 W4A16(4-bit 가중치)로 통일해 서로 비교 가능하게 한다.

    model_id가 EXAONE 계열이면 AWQ와 SpinQuant 모두에 수동 mappings를
    적용한다. 그 외(Solar 등 표준 Llama 아키텍처)는 llmcompressor의
    자동 추론에 맡긴다.
    """
    from llmcompressor.modifiers.awq import AWQModifier
    from llmcompressor.modifiers.quantization import GPTQModifier
    from llmcompressor.modifiers.transform import SpinQuantModifier

    is_exaone = model_id is not None and "EXAONE" in model_id.upper()

    if method == "awq":
        awq_kwargs = {"ignore": ["lm_head"], "scheme": "W4A16", "targets": ["Linear"]}
        if is_exaone:
            awq_kwargs["mappings"] = _exaone_awq_mappings()
        return [AWQModifier(**awq_kwargs)]

    if method == "gptq":
        return [GPTQModifier(targets="Linear", scheme="W4A16", ignore=["lm_head"])]

    if method in ("quarot", "spinquant"):
        learnable = (method == "spinquant")
        if learnable:
            # 조용히 실패해 깊은 pydantic 스택트레이스를 보게 하지 않고,
            # 원인을 바로 알 수 있는 메시지로 먼저 막는다.
            raise NotImplementedError(
                "spinquant(learnable=True)는 llmcompressor==0.10.0.3에서 아직 "
                "구현되어 있지 않다(라이브러리 자체의 validate_not_implemented). "
                "현재 이 환경에서는 quarot(learnable=False)만 실행 가능하다."
            )

        spinquant_kwargs = {
            "rotations": ["R1", "R2", "R3", "R4"],
            "transform_type": "hadamard",
            "learnable": learnable,
        }
        if is_exaone:
            # R3, R4 모두 제외한다. 이유는 두 가지가 겹친다.
            #
            # (1) 기술적 제약: R3(q_attn/k_cache 훅)은 EXAONE 3.5의 고정 리비전이
            #     transformers의 AttentionInterface를 지원하지 않아 구조적으로
            #     막힌다(AssertionError로 실측 확인). R4는 EXAONE의
            #     intermediate_size가 2의 거듭제곱이 아니라 결정적 Hadamard
            #     행렬을 만들 수 없다(ValueError로 실측 확인: "size != 2^n").
            #
            # (2) 방법론적 정당성 — 더 근본적인 이유다: R3와 R4는 모두 "같은
            #     텐서를 회전했다가 곧바로 역회전으로 상쇄"하는 방식으로
            #     동작한다(R4: c_proj 입력을 R로 회전 + c_proj 가중치를 R⁻¹로
            #     역회전 → x@R @ R⁻¹@W = x@W, 완전한 항등 연산). 이 상쇄가
            #     의미를 가지려면 회전 직후 그 지점에서 활성화를 저비트로
            #     양자화해야 하는데(W4A4/KV4 시나리오), 본 실험은 W4A16으로
            #     통일했으므로 활성화 양자화 자체가 없다. 즉 R3/R4는 기술적
            #     문제가 없었더라도 우리 실험 조건에서는 결과에 아무 영향을
            #     주지 않는 항등 연산이었을 것이다.
            #     반면 R1(residual stream)과 R2(어텐션 v/o)는 서로 다른 두
            #     가중치 행렬에 나눠 걸리므로 상쇄되지 않고 가중치 분포 자체를
            #     바꾼다 — W4A16에서도 실제 효과가 있다. 실측으로도 R1
            #     226/226, R2 64/64 매칭·적용이 두 번 다 문제없이 성공했다.
            spinquant_kwargs["rotations"] = ["R1", "R2"]

            import torch as _torch
            mapping, norm_mappings = _exaone_spinquant_mappings()
            head_dim = _exaone_head_dim(model_id, revision)
            transform_config = _build_exaone_transform_config(
                mapping, head_dim,
                rotations=spinquant_kwargs["rotations"],
                transform_type=spinquant_kwargs["transform_type"],
                precision=_torch.float64,
            )
            spinquant_kwargs["mappings"] = mapping
            spinquant_kwargs["norm_mappings"] = norm_mappings
            spinquant_kwargs["transform_config"] = transform_config

        # 회전 이후의 양자화는 GPTQModifier를 쓴다. QuaRot 원 논문의 구성이
        # "회전 + GPTQ"이기 때문이다. 초기 구현에서 QuantizationModifier(단순
        # round-to-nearest)를 쓴 결과, 캘리브레이션 데이터를 0개 사용하는
        # data-free 양자화가 수행되어(로그: "Inferred DataFreePipeline",
        # "Calibrating weights: 224/224 [00:00]") AWQ/GPTQ(각 256샘플로 오차 보정)
        # 와 공정하게 비교할 수 없었다. 실측 저하율도 KMMLU 19.43%, HAE-RAE
        # 26.91%로 비정상적으로 컸는데, 이는 회전의 효과가 아니라 오차 보정
        # 유무의 차이를 측정한 결과였다.
        return [
            SpinQuantModifier(**spinquant_kwargs),
            GPTQModifier(targets="Linear", scheme="W4A16", ignore=["lm_head"]),
        ]

    raise ValueError(f"알 수 없는 method: {method}")


def _pin_remote_code_files(model_id: str, revision: str, out_dir: Path) -> None:
    """
    model.save_pretrained()의 custom_object_save 메커니즘이 원격 코드(.py)를
    복사할 때 우리가 지정한 revision을 정확히 따라가지 않고, 이미 캐시에 있는
    다른(예: main의 최신) 리비전 파일을 덮어써버리는 사례가 실제로 확인됐다
    (EXAONE의 configuration_exaone.py가 transformers v5용 RopeParameters를
    쓰는 최신본으로 바뀌어, 우리 transformers==4.57.6과 다시 충돌했다).
    내부 동작을 추측으로 고치는 대신, 우리가 검증해둔 정확한 리비전의 .py
    파일을 HF 캐시에서 직접 찾아 출력 디렉터리에 강제로 덮어써 확실히 한다.
    """
    from huggingface_hub import snapshot_download
    log(f"[{revision[:8]}] 원격 코드 파일을 지정 리비전으로 고정하는 중...")
    snapshot_dir = Path(snapshot_download(
        repo_id=model_id, revision=revision, allow_patterns=["*.py"]
    ))
    copied = []
    for py_file in snapshot_dir.glob("*.py"):
        dest = out_dir / py_file.name
        dest.write_bytes(py_file.read_bytes())
        copied.append(py_file.name)
    log(f"  고정된 파일: {copied}")


def quantize_with_llmcompressor(model_id: str, output_dir: Path, method: str,
                                 calib_dataset, revision: str = None,
                                 calib_lang: str = DEFAULT_CALIB_LANG,
                                 calib_seed: int = 0,
                                 calib_n: int = NUM_CALIB_SAMPLES) -> str:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from llmcompressor import oneshot

    log(f"[{method}] llmcompressor 기반 4-bit 양자화 시작... "
        f"(revision={revision or 'main'}, calib_lang={calib_lang}, calib_seed={calib_seed})")
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, revision=revision, trust_remote_code=True
    )
    recipe = build_recipe(method, model_id=model_id, revision=revision)

    model = None
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, revision=revision, torch_dtype="auto", trust_remote_code=True
        )

        oneshot(
            model=model,
            processor=tokenizer,
            dataset=calib_dataset,
            recipe=recipe,
            max_seq_length=MAX_SEQ_LENGTH,
            num_calibration_samples=len(calib_dataset),
        )

        # 기본 언어(ko)는 기존 경로(<method>_4bit)를 그대로 유지해 이미 확보한
        # 결과와의 호환을 깨지 않고, 다른 언어만 접미사를 붙인다.
        suffix = "" if calib_lang == DEFAULT_CALIB_LANG else f"_calib_{calib_lang}"
        # seed 0은 접미사 없이 기존 경로를 그대로 쓴다(기존 결과 = seed 0).
        # 기본 크기(256)와 seed 0은 접미사 없이 기존 경로를 그대로 쓴다.
        if calib_n != NUM_CALIB_SAMPLES:
            suffix += f"_n{calib_n}"
        if calib_seed:
            suffix += f"_s{calib_seed}"
        out_dir = output_dir / f"{method}_4bit{suffix}"
        out_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(out_dir), save_compressed=True)
        tokenizer.save_pretrained(str(out_dir))

        if revision is not None:
            _pin_remote_code_files(model_id, revision, out_dir)

        log(f"[{method}] 저장 완료 → {out_dir}")
        return str(out_dir)
    finally:
        if model is not None:
            del model
        gc.collect()
        torch.cuda.empty_cache()


# ============================================================
# 효율성 측정
# ============================================================
class _CaptureAndReplayStdio:
    """
    OS 파일서술자(fd 1/2) 레벨에서 출력을 통째로 가로챈 뒤, 종료 시 그대로
    다시 출력해 터미널에서는 아주 짧게(수 초) 지연되는 것 외에 평소와
    동일하게 보이도록 한다. vLLM의 EngineCore는 별도 프로세스에서 로그를
    찍으므로, 파이썬 logging 모듈이 아니라 fd 레벨에서 가로채야 그 출력까지
    빠짐없이 잡을 수 있다.

    긴 oneshot() 호출 전체를 감싸면 실시간 모니터링이 불가능해지므로,
    반드시 LLM(...) 초기화처럼 짧게 끝나는 구간에만 사용한다.
    """

    def __enter__(self):
        sys.stdout.flush()
        sys.stderr.flush()
        self._saved_stdout_fd = os.dup(1)
        self._saved_stderr_fd = os.dup(2)
        self._tmp = tempfile.TemporaryFile(mode="w+b")
        os.dup2(self._tmp.fileno(), 1)
        os.dup2(self._tmp.fileno(), 2)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(self._saved_stdout_fd, 1)
        os.dup2(self._saved_stderr_fd, 2)
        os.close(self._saved_stdout_fd)
        os.close(self._saved_stderr_fd)
        self._tmp.seek(0)
        raw = self._tmp.read()
        self._tmp.close()
        self.text = raw.decode(errors="replace")
        # 가로챈 내용을 그대로 재출력해, 터미널에서는 짧게 지연된 것 외에
        # 평소와 동일한 실시간 로그 경험을 유지한다.
        sys.stdout.write(self.text)
        sys.stdout.flush()
        return False


def _parse_vllm_init_log(text: str) -> dict:
    """
    vLLM 초기화 로그에서 논문에 실제로 의미 있는 지표를 파싱한다.
    total_allocated_gb(모델+KV풀 총량)는 gpu_memory_utilization 비율만큼
    항상 비슷한 값으로 수렴하도록 설계되어 있어 기법 간 비교에 오도적이므로
    (실측: BF16 66.70GB vs AWQ 66.62GB — 모델 자체는 14.57GiB→4.93GiB로
    줄었는데도 총량은 거의 그대로였다), 아래 세 지표로 대체한다.
    - model_loading_gib: 가중치 자체가 실제로 차지한 메모리 (압축 효과의 직접 증거)
    - kv_cache_available_gib: 같은 GPU 예산에서 KV 캐시로 남은 메모리
    - kv_cache_tokens: 그 KV 캐시가 담을 수 있는 토큰 수
    - max_concurrency: 2048토큰 요청 기준 동시 처리 가능 배수
    패턴이 버전 차이 등으로 매칭되지 않으면 해당 값은 None으로 남기고
    조용히 넘어간다 — 파싱 실패가 실험 자체를 중단시켜서는 안 된다.
    """
    def _search(pattern, cast=float):
        m = re.search(pattern, text)
        if not m:
            return None
        val = m.group(1).replace(",", "")
        try:
            return cast(val)
        except ValueError:
            return None

    return {
        "model_loading_gib": _search(r"Model loading took ([\d.]+) Gi?B memory"),
        "kv_cache_available_gib": _search(r"Available KV cache memory: ([\d.]+) Gi?B"),
        "kv_cache_tokens": _search(r"GPU KV cache size: ([\d,]+) tokens", cast=int),
        "max_concurrency": _search(r"Maximum concurrency for [\d,]+ tokens per request: ([\d.]+)x"),
    }


def check_tokenizer_consistency(model_id: str, revision: str, quantized_path: str) -> dict:
    """
    양자화 저장 과정에서 tokenizer 설정 파일이 원본과 다르게 직렬화되어
    실제 토큰화 결과가 달라지는 경우를 잡기 위한 검증이다. AWQ 단계에서
    "incorrect regex pattern... fix_mistral_regex=True" 경고가 관찰되어
    (원인은 tokenizer 재직렬화 과정으로 추정되며 확인이 필요하다), 결과가
    실제로 달라지는지 자동으로 확인해 results.json에 남긴다.
    """
    from transformers import AutoTokenizer

    orig_tok = AutoTokenizer.from_pretrained(
        model_id, revision=revision, trust_remote_code=True
    )
    quant_tok = AutoTokenizer.from_pretrained(quantized_path, trust_remote_code=True)

    test_texts = get_prompts(BATCH_SIZE)
    mismatches = []
    for text in test_texts:
        orig_ids = orig_tok(text)["input_ids"]
        quant_ids = quant_tok(text)["input_ids"]
        if orig_ids != quant_ids:
            mismatches.append({
                "text": text,
                "orig_len": len(orig_ids),
                "quant_len": len(quant_ids),
            })

    result = {
        "num_checked": len(test_texts),
        "num_mismatched": len(mismatches),
        "mismatches": mismatches[:5],  # 전부 남기면 파일이 불필요하게 커지므로 일부만
    }
    if mismatches:
        log(f"  경고: tokenizer 불일치 {len(mismatches)}/{len(test_texts)}건 발견 "
            f"— results.json의 tokenizer_check 참고")
    else:
        log(f"  tokenizer 일관성 확인 완료: {len(test_texts)}개 문장 전부 일치")
    return result


def measure_efficiency_vllm(model_path: str, method_name: str, is_bf16: bool = False,
                             revision: str = None) -> dict:
    """
    quantization 인자를 별도로 지정하지 않는다 — llmcompressor가 저장한
    compressed-tensors 포맷은 config.json에 양자화 방식이 기록되어 있어
    vLLM이 자동으로 인식한다. AWQ/GPTQ/QuaRot/SpinQuant 네 가지 산출물
    모두 동일한 로딩 경로를 탄다.

    revision은 bf16 단계처럼 model_path가 HF Hub의 원본 저장소 ID일 때만
    의미가 있다 — 양자화 산출물(model_path가 로컬 디렉터리)에는 적용되지 않는다.
    """
    from vllm import LLM, SamplingParams

    log(f"[{method_name}] 효율성 측정 시작 (vLLM, revision={revision or 'main'})")
    dtype = "bfloat16" if is_bf16 else "auto"
    llm_kwargs = {
        "model": model_path,
        "trust_remote_code": True,
        "dtype": dtype,
        "gpu_memory_utilization": 0.70,
        "max_model_len": 2048,
    }
    if revision is not None:
        llm_kwargs["revision"] = revision
        llm_kwargs["tokenizer_revision"] = revision

    llm = None
    try:
        baseline_gb = get_gpu_memory_used_gb()

        # LLM(...) 초기화 구간만 짧게 가로챈다 — 이 안에서 vLLM이
        # "Model loading took X GiB", "GPU KV cache size: X tokens" 등을
        # 로그로 찍으므로, 그 텍스트를 파싱해 논문에 실제로 쓸 지표를 뽑는다.
        with _CaptureAndReplayStdio() as cap:
            llm = LLM(**llm_kwargs)
        vllm_init_metrics = _parse_vllm_init_log(cap.text)

        # 엔진 초기화 직후 총 점유량 (모델 + KV 캐시 풀). EngineCore 서브프로세스가
        # 실제로 GPU에 올린 양을 pynvml로 프로세스 외부에서 직접 읽는다 — vLLM V1의
        # 별도 프로세스 구조 때문에 torch.cuda 기준 측정은 항상 0에 가깝게 나온다.
        # 주의: 이 값은 gpu_memory_utilization 비율만큼 항상 비슷하게 수렴하도록
        # 설계되어 있어(남는 공간을 KV 캐시가 흡수), 기법 간 비교에는 부적합하다 —
        # 실측: BF16 66.70GB vs AWQ 66.62GB(모델 자체는 14.57GiB→4.93GiB로 줄었음).
        # 실제 압축 효과 비교에는 vllm_init_metrics의 model_loading_gib를 쓸 것.
        after_init_gb = get_gpu_memory_used_gb()
        total_allocated_gb = after_init_gb - baseline_gb
        log(f"  모델+KV풀 총 점유량(참고용, 기법간 비교 부적합): {total_allocated_gb:.2f} GB")
        log(f"  실제 압축 지표: model_loading={vllm_init_metrics['model_loading_gib']} GiB, "
            f"kv_cache_tokens={vllm_init_metrics['kv_cache_tokens']}, "
            f"max_concurrency={vllm_init_metrics['max_concurrency']}x")

        prompts = get_prompts(BATCH_SIZE)
        sampling_params = SamplingParams(temperature=0.0, max_tokens=MAX_NEW_TOKENS)

        for _ in range(NUM_WARMUP):
            _ = llm.generate(prompts, sampling_params)

        # 참고: vLLM은 gpu_memory_utilization 비율만큼 KV 캐시 풀을 엔진 초기화
        # 시점에 미리 통째로 확보해두므로, generate() 호출마다 GPU 메모리가 추가로
        # 늘어나는 전통적인 의미의 "런별 peak"는 애초에 관측되지 않는다. 아래
        # run_used_gb는 각 실행 시점의 총 점유량을 재확인하는 용도이며, 값이
        # total_allocated_gb와 거의 같게 나오는 것이 정상이다.
        run_mems, tps_list = [], []
        for run in range(NUM_RUNS):
            start = time.time()
            outputs = llm.generate(prompts, sampling_params)
            elapsed = time.time() - start

            run_used_gb = get_gpu_memory_used_gb() - baseline_gb
            total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
            tps = total_tokens / max(elapsed, 1e-6)

            run_mems.append(run_used_gb)
            tps_list.append(tps)
            log(f"  Run {run + 1}: gpu_used={run_used_gb:.2f} GB, {tps:.1f} tok/s")

        return {
            "method": method_name,
            "total_allocated_gb": round(total_allocated_gb, 2),
            "total_allocated_gb_note": (
                "gpu_memory_utilization 비율만큼 항상 비슷하게 수렴하는 값 — "
                "기법 간 압축 효과 비교에는 vllm_init_metrics를 사용할 것"
            ),
            "vllm_init_metrics": vllm_init_metrics,
            "run_used_gb_mean": round(float(np.mean(run_mems)), 2),
            "run_used_gb_std": round(float(np.std(run_mems)), 2),
            "throughput_tps_mean": round(float(np.mean(tps_list)), 2),
            "throughput_tps_std": round(float(np.std(tps_list)), 2),
            "batch_size": len(prompts),
            "max_new_tokens": MAX_NEW_TOKENS,
            "num_runs": NUM_RUNS,
            "engine": "vllm",
            "memory_measurement": "pynvml (process-external)",
        }
    finally:
        if llm is not None:
            del llm
        gc.collect()
        torch.cuda.empty_cache()


# ============================================================
# 평가 스크립트 생성
# ============================================================
def generate_eval_script(model_id: str, output_dir: Path, method_paths: dict,
                          revision: str = None) -> Path:
    """
    --model vllm 백엔드를 사용한다. kmmlu 하나가 수십~수백 개 세부 과목
    서브태스크를 묶은 집계 태스크라 --model hf(순수 transformers generate)로는
    체감 소요 시간이 크게 늘어난다. 우리 venv에 이미 vLLM이 설치되어 있으므로
    같은 엔진으로 평가도 수행해 시간을 아낀다.
    """
    lines = ["#!/bin/bash", "set -e", ""]

    def _block(label, model_arg, out_name):
        return [
            f'echo "=== {label} ==="',
            "lm_eval --model vllm \\",
            f"  --model_args pretrained={model_arg},trust_remote_code=True,"
            f"gpu_memory_utilization=0.70,max_model_len=2048 \\",
            f"  --tasks {EVAL_TASKS} --num_fewshot 5 --batch_size auto \\",
            f"  --output_path {output_dir}/eval_{out_name}",
            "",
        ]

    bf16_model_arg = f"{model_id},dtype=bfloat16"
    if revision is not None:
        # BF16은 model_id를 HF Hub에서 직접 로드하므로, 우리가 고정한 리비전을
        # 평가 스크립트에도 그대로 반영해야 두 실행이 같은 config로 비교된다.
        bf16_model_arg += f",revision={revision}"
    lines += _block("BF16", bf16_model_arg, "bf16")
    for method_name, out_name in [
        ("AWQ-4bit", "awq"),
        ("GPTQ-4bit", "gptq"),
        ("QuaRot-4bit", "quarot"),
        ("SpinQuant-4bit", "spinquant"),
    ]:
        path = method_paths.get(method_name)
        if path:
            lines += _block(method_name, path, out_name)

    sh_path = output_dir / "run_eval.sh"
    with open(sh_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    os.chmod(sh_path, 0o755)
    log(f"평가 스크립트 생성 → {sh_path}")
    return sh_path


# ============================================================
# 단계 실행기
# ============================================================
def run_stage(stage: str, model_id: str, output_dir: Path, revision: str = None, calib_seed: int = 0,
              calib_n: int = NUM_CALIB_SAMPLES,
               calib_lang: str = DEFAULT_CALIB_LANG) -> None:
    results_file = output_dir / "results.json"
    results = load_or_init_results(results_file)
    env_info = get_environment_info(model_id)
    env_info["model_revision"] = revision or "main"
    env_info["calib_lang"] = calib_lang
    env_info["calib_seed"] = calib_seed
    env_info["calib_n"] = calib_n
    _cfg = CALIB_DATASETS[calib_lang]
    env_info["calib_dataset"] = (
        f"mixed({'+'.join(_cfg['mix'])})" if "mix" in _cfg else _cfg["path"]
    )
    results["environment"] = env_info
    save_results(results_file, results)

    # 기본 언어(ko)는 기존 이름을 유지하고, 다른 언어만 접미사를 붙여
    # results.json 안에서 서로 덮어쓰지 않게 한다.
    method_name = METHOD_NAMES[stage]
    if stage != "bf16" and calib_lang != DEFAULT_CALIB_LANG:
        method_name = f"{method_name}[calib:{calib_lang}]"
    if stage != "bf16" and calib_n != NUM_CALIB_SAMPLES:
        method_name = f"{method_name}[n:{calib_n}]"
    if stage != "bf16" and calib_seed:
        method_name = f"{method_name}[seed:{calib_seed}]"

    try:
        if stage == "bf16":
            eff = measure_efficiency_vllm(model_id, method_name, is_bf16=True, revision=revision)
        else:
            manifest, provenance = [], {}
            calib_dataset = load_calib_dataset(calib_lang, calib_n, seed=calib_seed,
                                               manifest=manifest, provenance=provenance)
            manifest_path = save_calib_manifest(output_dir / "calib_manifests", calib_lang,
                                                calib_n, calib_seed, manifest, provenance)
            out_path = quantize_with_llmcompressor(
                model_id, output_dir, stage, calib_dataset,
                revision=revision, calib_lang=calib_lang, calib_seed=calib_seed,
                calib_n=calib_n
            )
            # 양자화 산출물은 로컬 디렉터리이므로 이 이후로는 revision이 필요 없다.
            eff = measure_efficiency_vllm(out_path, method_name)
            eff["model_path"] = out_path
            eff["calib_lang"] = calib_lang
            eff["calib_seed"] = calib_seed
            eff["calib_manifest"] = str(manifest_path)
            eff["tokenizer_check"] = check_tokenizer_consistency(model_id, revision, out_path)

        results["efficiency"] = [r for r in results["efficiency"] if r["method"] != method_name]
        results["efficiency"].append(eff)
        save_results(results_file, results)

    except Exception as e:
        log(f"[{stage}] 실패: {e}")
        traceback.print_exc()
        results["errors"].append({
            "stage": stage,
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
        })
        save_results(results_file, results)
        del e
        gc.collect()
        torch.cuda.empty_cache()
        sys.exit(1)

    method_paths = {
        r["method"]: r.get("model_path")
        for r in results["efficiency"]
        if r.get("model_path")
    }
    generate_eval_script(model_id, output_dir, method_paths, revision=revision)
    log(f"[{stage}] 단계 완료")


def parse_args():
    parser = argparse.ArgumentParser(description="단일 venv 통합형 양자화 비교 실험")
    parser.add_argument("--stage", required=True, choices=STAGES)
    parser.add_argument("--model", default=DEFAULT_MODEL_ID,
                         help="HuggingFace 모델 ID (예: upstage/SOLAR-10.7B-Instruct-v1.0)")
    parser.add_argument("--revision", default=None,
                         help="HF Hub 리비전(커밋 해시). 지정하지 않으면 --model이 "
                              "DEFAULT_MODEL_ID일 때만 DEFAULT_REVISION이 자동 적용된다. "
                              "다른 모델(예: Solar)을 쓸 때는 그 저장소에 맞는 리비전을 "
                              "직접 확인해 지정할 것.")
    parser.add_argument("--calib-n", type=int, default=NUM_CALIB_SAMPLES,
                         help="캘리브레이션 표본 수(기본 256). 256이 아니면 산출물 경로에 "
                              "_n<수> 접미사가 붙는다.")
    parser.add_argument("--calib-seed", type=int, default=0,
                         help="캘리브레이션 샘플 추출 시드. 0은 데이터셋 앞에서부터 결정적으로 "
                              "추출(시드 도입 전 실험과 동일). 1 이상은 셔플 후 추출하며 "
                              "산출물 경로에 _s<seed> 접미사가 붙는다.")
    parser.add_argument("--calib-lang", default=DEFAULT_CALIB_LANG,
                         choices=list(CALIB_DATASETS.keys()),
                         help="캘리브레이션 데이터 언어. ko=KoAlpaca(기본), en=C4 영어. "
                              "언어별 ablation 시 사용한다. 기본값(ko) 외의 언어로 실행하면 "
                              "산출물이 <method>_4bit_calib_<lang>/ 에 별도로 저장되어 "
                              "기존 결과를 덮어쓰지 않는다.")
    return parser.parse_args()


def main():
    args = parse_args()
    enforce_cuda_visible_devices()

    revision = args.revision
    if revision is None and args.model == DEFAULT_MODEL_ID:
        revision = DEFAULT_REVISION

    output_dir = Path("./quant_experiments") / args.model.split("/")[-1]
    output_dir.mkdir(parents=True, exist_ok=True)

    log("=" * 70)
    log(f"단계 실행: stage={args.stage}, model={args.model}, "
        f"revision={revision or 'main'}, calib_lang={args.calib_lang}")
    log("=" * 70)

    if not check_gpu():
        log("GPU 상태 점검 실패로 종료합니다.")
        sys.exit(1)

    run_stage(args.stage, args.model, output_dir, revision=revision,
              calib_lang=args.calib_lang, calib_seed=args.calib_seed, calib_n=args.calib_n)


if __name__ == "__main__":
    main()
