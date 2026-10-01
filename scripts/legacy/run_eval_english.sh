#!/bin/bash
# 실험 1: 영어 벤치마크 평가 (언어별 열화 비대칭 분석용)
#
# 목적: 같은 4비트 체크포인트를 영어 벤치마크로도 평가해, 한국어 열화 폭과
#       비교한다. KMMLU↔MMLU, KoBEST HellaSwag↔HellaSwag처럼 대응 관계가
#       있는 태스크 쌍을 사용해 난이도를 통제한다.
#
# 사용법:
#   cd /path/to/quant_exp
#   source venv-quant/bin/activate
#   export CUDA_VISIBLE_DEVICES=0
#   export HF_HOME=/path/to/.cache/huggingface
#   nohup bash run_eval_english.sh > eval_english_log.txt 2>&1 &
#   disown

set -u   # 미정의 변수 사용 시 오류. set -e는 쓰지 않는다 —
         # 한 기법이 실패해도 나머지는 계속 평가하기 위함.

if [ -z "${VIRTUAL_ENV:-}" ]; then
  echo "오류: venv가 활성화되어 있지 않다. source venv-quant/bin/activate 먼저 실행할 것."
  exit 1
fi
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  echo "오류: CUDA_VISIBLE_DEVICES가 설정되어 있지 않다. export CUDA_VISIBLE_DEVICES=0 먼저 실행할 것."
  exit 1
fi

BASE="quant_experiments/EXAONE-3.5-7.8B-Instruct"
MODEL_ID="LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct"
REVISION="0ff6b5ec7c13b049b253a16a889aa269e6b79a94"

# 한국어 태스크와 대응되는 영어 태스크.
#   mmlu          <-> kmmlu          (다지선다 지식)
#   hellaswag     <-> kobest_hellaswag (상식적 문장 완성)
#   arc_challenge  : 추가 추론 태스크
#   boolq         <-> kobest_boolq   (참/거짓 추론)
# 주의: 태스크명이 설치된 lm-evaluation-harness 버전에 실제로 존재하는지
#       사전 확인할 것 (TaskManager().all_tasks 로 조회).
EVAL_TASKS_EN="mmlu,hellaswag,arc_challenge,boolq"

COMMON_ARGS="trust_remote_code=True,gpu_memory_utilization=0.70,max_model_len=2048"

run_eval () {
  local label="$1"
  local pretrained="$2"
  local out_name="$3"
  echo "=== ${label} (English) ==="
  date
  HF_HUB_OFFLINE=1 lm_eval --model vllm \
    --model_args "pretrained=${pretrained},${COMMON_ARGS}" \
    --tasks "${EVAL_TASKS_EN}" --num_fewshot 5 --batch_size auto \
    --output_path "${BASE}/eval_en_${out_name}"
  echo "--- ${label} 종료 (exit=$?) ---"
  date
}

# BF16은 HF Hub에서 직접 로드하므로 리비전을 함께 고정한다.
run_eval "BF16"              "${MODEL_ID},dtype=bfloat16,revision=${REVISION}" "bf16"
run_eval "AWQ-4bit"          "${BASE}/awq_4bit"                                "awq"
run_eval "GPTQ-4bit"         "${BASE}/gptq_4bit"                               "gptq"
run_eval "QuaRot-4bit(GPTQ)" "${BASE}/quarot_4bit"                             "quarot"
run_eval "QuaRot-4bit(RTN)"  "${BASE}/quarot_4bit_rtn_backup"                  "quarot_rtn_backup"

echo "모든 영어 벤치마크 평가 완료."
date
