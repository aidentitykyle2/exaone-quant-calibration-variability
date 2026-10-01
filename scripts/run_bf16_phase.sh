#!/bin/bash
# run_bf16_phase.sh — 32B BF16 기준선 평가 (한국어·영어 동시)
#
# 32B BF16은 가중치만 약 64GB다. 단일 H100(93GB)에 올리려면
# gpu_memory_utilization을 0.90 이상으로 올려야 하며, 그래도 KV 캐시 여유가
# 크지 않다. 4장을 쓸 수 있으므로 텐서 병렬(TP=2)로 두 장씩 묶어
# 한국어·영어 평가를 동시에 수행하는 편이 안전하고 빠르다.
#
#   GPU 0,1 → 한국어 평가 (TP=2)
#   GPU 2,3 → 영어 평가   (TP=2)
#
# MODE=single 로 설정하면 TP 없이 GPU 한 장씩 사용한다(메모리 여유 확인용).
#
# 사용법:
#   cd /path/to/quant_exp
#   source venv-quant/bin/activate
#   export HF_HOME=/path/to/.cache/huggingface
#   nohup bash run_bf16_phase.sh > bf16_phase.txt 2>&1 & disown

set -u

if [ -z "${VIRTUAL_ENV:-}" ]; then
  echo "오류: venv 미활성화. source venv-quant/bin/activate 먼저 실행할 것."
  exit 1
fi

MODEL_ID="${MODEL_ID:-LGAI-EXAONE/EXAONE-4.0-32B}"
MODEL_SHORT="$(basename "${MODEL_ID}")"
BASE="quant_experiments/${MODEL_SHORT}"
MODE="${MODE:-tp2}"
# 원격 코드가 갱신되면 구버전 transformers에서 깨지므로 리비전을 고정한다.
REVISION="${REVISION-0ff6b5ec7c13b049b253a16a889aa269e6b79a94}"
REV_ARG=""
[ -n "${REVISION}" ] && REV_ARG=",revision=${REVISION}"

# EXAONE 4.0 검증 실험처럼 핵심 두 벤치마크만 필요할 때 환경변수로 줄인다.
TASKS_KO="${TASKS_KO:-kmmlu,kobest_boolq,kobest_copa,kobest_wic,kobest_hellaswag,kobest_sentineg,haerae}"
TASKS_EN="${TASKS_EN:-mmlu,hellaswag,arc_challenge,boolq}"
GPU_UTIL="${GPU_UTIL:-0.90}"

# GPU 배치. 환경변수로 덮어쓸 수 있다.
#   tp2    : 두 장을 텐서 병렬로 묶어 한 평가를 수행 (32B BF16에 안전)
#   single : 한 장에 모델 전체를 올림 (gpu_memory_utilization 0.90 필요, 실패 가능)
#   seq    : GPU 한 장으로 한국어→영어 순차 (가용 GPU가 적을 때)
if [ "${MODE}" = "tp2" ]; then
  KO_GPUS="${KO_GPUS:-0,2}"; EN_GPUS="${EN_GPUS:-3}"; TP_KO=2; TP_EN=1
elif [ "${MODE}" = "seq" ]; then
  KO_GPUS="${KO_GPUS:-0,2}"; EN_GPUS="${EN_GPUS:-0,2}"; TP_KO=2; TP_EN=2
else
  KO_GPUS="${KO_GPUS:-0}";   EN_GPUS="${EN_GPUS:-2}"; TP_KO=1; TP_EN=1
fi

ARGS_KO="pretrained=${MODEL_ID}${REV_ARG},dtype=bfloat16,trust_remote_code=True,gpu_memory_utilization=${GPU_UTIL},max_model_len=2048,tensor_parallel_size=${TP_KO}"
ARGS_EN="pretrained=${MODEL_ID}${REV_ARG},dtype=bfloat16,trust_remote_code=True,gpu_memory_utilization=${GPU_UTIL},max_model_len=2048,tensor_parallel_size=${TP_EN}"

echo "############ BF16 기준선 평가 (MODE=${MODE}) ############"
echo "한국어 → GPU ${KO_GPUS} / 영어 → GPU ${EN_GPUS}"
date

echo "=== BF16 한국어 평가 시작 ==="
CUDA_VISIBLE_DEVICES="${KO_GPUS}" HF_HUB_OFFLINE=1 lm_eval --model vllm \
  --model_args "${ARGS_KO}" \
  --tasks "${TASKS_KO}" --num_fewshot 5 --batch_size auto \
  --log_samples \
  --output_path "${BASE}/seed_eval/ko/bf16" \
  > bf16_eval_ko.txt 2>&1 &
PID_KO=$!

# 두 프로세스가 동시에 64GB 가중치를 읽지 않도록 간격을 둔다.
sleep 180

if [ "${MODE}" = "seq" ]; then
  echo "=== seq 모드: 한국어 완료를 기다린 뒤 영어 평가 ==="
  wait "${PID_KO}"; KO_EXIT=$?
  echo "--- BF16 한국어 종료 (exit=${KO_EXIT}) ---"
fi

echo "=== BF16 영어 평가 시작 ==="
CUDA_VISIBLE_DEVICES="${EN_GPUS}" lm_eval --model vllm \
  --model_args "${ARGS_EN}" \
  --tasks "${TASKS_EN}" --num_fewshot 5 --batch_size auto \
  --log_samples \
  --output_path "${BASE}/seed_eval/en/bf16" \
  > bf16_eval_en.txt 2>&1 &
PID_EN=$!

echo "한국어 PID=${PID_KO} (로그: bf16_eval_ko.txt)"
echo "영어   PID=${PID_EN} (로그: bf16_eval_en.txt)"

if [ "${MODE}" != "seq" ]; then
  wait "${PID_KO}"; KO_EXIT=$?
  echo "--- BF16 한국어 종료 (exit=${KO_EXIT}) ---"
fi
wait "${PID_EN}"; EN_EXIT=$?
echo "--- BF16 영어 종료 (exit=${EN_EXIT}) ---"
date
echo "############ BF16 단계 완료 ############"
