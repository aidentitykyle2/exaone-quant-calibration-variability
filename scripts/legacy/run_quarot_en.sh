#!/bin/bash
# 실험 D: 회전 × 캘리브레이션 언어 2x2 설계 완성
#
# 목적: QuaRot(회전+GPTQ)을 영어 캘리브레이션으로 재양자화해, 아래 표의
#       빈칸을 채운다. 이것이 채워지면 "회전이 캘리브레이션 언어 선택에 대한
#       강건성을 높이는가"를 직접 검증할 수 있다.
#
#                  | calib:ko | calib:en
#       -----------|----------|----------
#       회전 없음   |   확보   |   확보
#       회전 있음   |   확보   |  이 실험
#
#   핵심 비교: ko→en 전환 시 한국어 성능 변동 폭을
#              GPTQ(1.55%→2.64%, +1.09%p)와 QuaRot에서 비교한다.
#              QuaRot의 변동이 더 작다면 회전이 언어 의존성을 완충한다는 근거가 된다.
#
# 구성: 양자화는 병렬화가 불가능하므로 GPU 하나로 수행하고,
#       이후 한국어/영어 평가를 두 GPU에서 동시에 실행해 시간을 절반으로 줄인다.
#       예상: 양자화 약 1시간 + 평가 약 1시간 = 총 2시간
#
# 사용법:
#   cd /path/to/quant_exp
#   source venv-quant/bin/activate
#   export HF_HOME=/path/to/.cache/huggingface
#   GPU_QUANT=0 GPU_A=0 GPU_B=3 nohup bash run_quarot_en.sh > quarot_en_log.txt 2>&1 &
#   disown
#
#   (CUDA_VISIBLE_DEVICES는 이 스크립트가 각 명령마다 직접 지정하므로
#    바깥에서 export할 필요가 없다.)

set -u

if [ -z "${VIRTUAL_ENV:-}" ]; then
  echo "오류: venv 미활성화. source venv-quant/bin/activate 먼저 실행할 것."
  exit 1
fi

GPU_QUANT="${GPU_QUANT:-0}"   # 양자화에 사용할 GPU
GPU_A="${GPU_A:-0}"           # 한국어 평가용 GPU
GPU_B="${GPU_B:-3}"           # 영어 평가용 GPU

BASE="quant_experiments/EXAONE-3.5-7.8B-Instruct"
MODEL_ID="LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct"
CKPT="${BASE}/quarot_4bit_calib_en"

TASKS_KO="kmmlu,kobest_boolq,kobest_copa,kobest_wic,kobest_hellaswag,kobest_sentineg,haerae"
TASKS_EN="mmlu,hellaswag,arc_challenge,boolq"
COMMON_ARGS="trust_remote_code=True,gpu_memory_utilization=0.70,max_model_len=2048"

echo "############ 1단계: QuaRot 재양자화 (calib=en, GPU ${GPU_QUANT}) ############"
date
CUDA_VISIBLE_DEVICES="${GPU_QUANT}" \
  python run_quant_stage.py --stage quarot --model "${MODEL_ID}" --calib-lang en
QUANT_EXIT=$?
echo "--- 양자화 종료 (exit=${QUANT_EXIT}) ---"
date

if [ "${QUANT_EXIT}" -ne 0 ]; then
  echo "양자화가 실패해 평가를 건너뛴다. 로그 상단의 오류를 확인할 것."
  exit 1
fi
if [ ! -d "${CKPT}" ]; then
  echo "오류: 체크포인트 디렉터리가 없다 → ${CKPT}"
  exit 1
fi

echo "############ 2단계: 한국어/영어 평가 동시 실행 ############"
echo "  한국어 → GPU ${GPU_A} / 영어 → GPU ${GPU_B}"
date

CUDA_VISIBLE_DEVICES="${GPU_A}" HF_HUB_OFFLINE=1 lm_eval --model vllm \
  --model_args "pretrained=${CKPT},${COMMON_ARGS}" \
  --tasks "${TASKS_KO}" --num_fewshot 5 --batch_size auto \
  --output_path "${BASE}/eval_quarot_calib_en" \
  > quarot_en_eval_ko.txt 2>&1 &
PID_KO=$!

# 두 프로세스가 동시에 모델을 적재하며 디스크/CPU 경합을 일으키지 않도록 간격을 둔다.
sleep 60

CUDA_VISIBLE_DEVICES="${GPU_B}" HF_HUB_OFFLINE=1 lm_eval --model vllm \
  --model_args "pretrained=${CKPT},${COMMON_ARGS}" \
  --tasks "${TASKS_EN}" --num_fewshot 5 --batch_size auto \
  --output_path "${BASE}/eval_en_quarot_calib_en" \
  > quarot_en_eval_en.txt 2>&1 &
PID_EN=$!

echo "한국어 평가 PID=${PID_KO} (로그: quarot_en_eval_ko.txt)"
echo "영어 평가 PID=${PID_EN} (로그: quarot_en_eval_en.txt)"

wait "${PID_KO}"; KO_EXIT=$?
wait "${PID_EN}"; EN_EXIT=$?
echo "--- 한국어 평가 종료 (exit=${KO_EXIT}) ---"
echo "--- 영어 평가 종료 (exit=${EN_EXIT}) ---"
date

echo "############ 3단계: 결과 취합 ############"
python compare_languages.py --output-dir "${BASE}"

echo "############ 전체 완료 ############"
date
