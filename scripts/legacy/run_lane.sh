#!/bin/bash
# run_lane.sh — 하나의 GPU에서 (양자화 → 한국어 평가 → 영어 평가)를 순차 수행
#
# 설계 의도:
#   llmcompressor의 SequentialPipeline은 레이어 단위로 처리하며 오프로딩하므로
#   양자화 시 GPU 점유가 모델 전체 크기보다 작다(7.8B 실측 약 11.9GB).
#   32B도 단일 H100(93GB)에 수용 가능할 것으로 추정되므로, 조건마다 GPU 하나를
#   배정해 4개 레인을 완전히 독립적으로 병렬 실행한다.
#
#   레인 간에는 의존 관계가 없다. 각 레인은 자기 체크포인트만 읽고 쓰며,
#   결과도 서로 다른 디렉터리에 저장된다.
#
# 사용법:
#   bash run_lane.sh <GPU번호> <METHOD> <CALIB_LANG> [MODEL_ID]
# 예:
#   bash run_lane.sh 0 awq  ko
#   bash run_lane.sh 2 gptq en
#
# 백그라운드 실행 권장:
#   nohup bash run_lane.sh 0 awq ko > lane_awq_ko.txt 2>&1 & disown

set -u

if [ "$#" -lt 3 ]; then
  echo "사용법: bash run_lane.sh <GPU번호> <METHOD> <CALIB_LANG> [MODEL_ID]"
  exit 1
fi

GPU="$1"
METHOD="$2"
CALIB="$3"
MODEL_ID="${4:-LGAI-EXAONE/EXAONE-4.0-32B}"

if [ -z "${VIRTUAL_ENV:-}" ]; then
  echo "오류: venv 미활성화. source venv-quant/bin/activate 먼저 실행할 것."
  exit 1
fi

MODEL_SHORT="$(basename "${MODEL_ID}")"
BASE="quant_experiments/${MODEL_SHORT}"

# 기본 언어(ko)는 접미사 없이 저장되는 run_quant_stage.py의 규칙을 따른다.
if [ "${CALIB}" = "ko" ]; then
  CKPT="${BASE}/${METHOD}_4bit"
  EVAL_KO="${BASE}/eval_${METHOD}"
  EVAL_EN="${BASE}/eval_en_${METHOD}"
else
  CKPT="${BASE}/${METHOD}_4bit_calib_${CALIB}"
  EVAL_KO="${BASE}/eval_${METHOD}_calib_${CALIB}"
  EVAL_EN="${BASE}/eval_en_${METHOD}_calib_${CALIB}"
fi

TASKS_KO="kmmlu,kobest_boolq,kobest_copa,kobest_wic,kobest_hellaswag,kobest_sentineg,haerae"
TASKS_EN="mmlu,hellaswag,arc_challenge,boolq"
# 32B는 BF16 가중치만 약 64GB다. 4비트 산출물(약 18~20GB)에도 동일한 값을 적용해
# KV 캐시 조건을 통일한다(7.8B 실험의 0.70에서 상향).
GPU_UTIL="${GPU_UTIL:-0.90}"
COMMON_ARGS="trust_remote_code=True,gpu_memory_utilization=${GPU_UTIL},max_model_len=2048"

echo "############################################################"
echo "# LANE 시작: GPU=${GPU} METHOD=${METHOD} CALIB=${CALIB}"
echo "# MODEL=${MODEL_ID}"
echo "# CKPT=${CKPT}"
echo "############################################################"
date

# ---------- 1단계: 양자화 ----------
echo "=== [${METHOD}/${CALIB}] 양자화 시작 ==="
date
CUDA_VISIBLE_DEVICES="${GPU}" \
  python run_quant_stage.py --stage "${METHOD}" --model "${MODEL_ID}" --calib-lang "${CALIB}"
QEXIT=$?
echo "--- [${METHOD}/${CALIB}] 양자화 종료 (exit=${QEXIT}) ---"
date

if [ "${QEXIT}" -ne 0 ]; then
  echo "양자화 실패로 이 레인을 중단한다. 다른 레인에는 영향이 없다."
  exit 1
fi
if [ ! -d "${CKPT}" ]; then
  echo "오류: 체크포인트가 생성되지 않았다 → ${CKPT}"
  exit 1
fi

# ---------- 2단계: 한국어 평가 ----------
echo "=== [${METHOD}/${CALIB}] 한국어 평가 시작 ==="
date
CUDA_VISIBLE_DEVICES="${GPU}" HF_HUB_OFFLINE=1 lm_eval --model vllm \
  --model_args "pretrained=${CKPT},${COMMON_ARGS}" \
  --tasks "${TASKS_KO}" --num_fewshot 5 --batch_size auto \
  --log_samples \
  --output_path "${EVAL_KO}"
echo "--- [${METHOD}/${CALIB}] 한국어 평가 종료 (exit=$?) ---"
date

# ---------- 3단계: 영어 평가 ----------
echo "=== [${METHOD}/${CALIB}] 영어 평가 시작 ==="
date
CUDA_VISIBLE_DEVICES="${GPU}" lm_eval --model vllm \
  --model_args "pretrained=${CKPT},${COMMON_ARGS}" \
  --tasks "${TASKS_EN}" --num_fewshot 5 --batch_size auto \
  --log_samples \
  --output_path "${EVAL_EN}"
echo "--- [${METHOD}/${CALIB}] 영어 평가 종료 (exit=$?) ---"
date

echo "############ LANE 완료: ${METHOD}/${CALIB} ############"
date
