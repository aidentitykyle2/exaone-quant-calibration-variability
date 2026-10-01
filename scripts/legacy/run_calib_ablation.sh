#!/bin/bash
# 실험 C: 캘리브레이션 언어 ablation
#
# 연구 가설(H1): 캘리브레이션 데이터의 언어가 양자화 후 언어별 성능 보존에
#                영향을 미친다.
#
# 설계: 양자화 알고리즘(AWQ, GPTQ)과 나머지 모든 조건(256샘플, 최소 길이,
#       max_seq_len, W4A16 스킴)을 고정하고 캘리브레이션 데이터의 언어만
#       바꾼다. 이렇게 해야 "알고리즘 차이"가 교란변수로 섞이지 않는다.
#       (기존 QuaRot(RTN) 관측은 캘리브레이션 유무와 알고리즘이 동시에
#        달랐으므로 인과 추론이 불가능했다.)
#
#   캘리브레이션 | 한국어 평가 | 영어 평가
#   ------------|-----------|----------
#   ko (기존)    |    확보됨   |   확보됨
#   en          |   이 실험   |  이 실험
#   mixed       |   이 실험   |  이 실험
#
# 사용법:
#   cd /path/to/quant_exp
#   source venv-quant/bin/activate
#   export CUDA_VISIBLE_DEVICES=3      # nvidia-smi로 빈 GPU 확인 후 지정
#   export HF_HOME=/path/to/.cache/huggingface
#   nohup bash run_calib_ablation.sh > calib_ablation_log.txt 2>&1 &
#   disown

set -u   # set -e는 쓰지 않는다 — 한 조건이 실패해도 나머지는 계속 진행.

if [ -z "${VIRTUAL_ENV:-}" ]; then
  echo "오류: venv 미활성화. source venv-quant/bin/activate 먼저 실행할 것."
  exit 1
fi
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  echo "오류: CUDA_VISIBLE_DEVICES 미설정. export CUDA_VISIBLE_DEVICES=<번호> 먼저 실행할 것."
  exit 1
fi

BASE="quant_experiments/EXAONE-3.5-7.8B-Instruct"
MODEL_ID="LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct"

TASKS_KO="kmmlu,kobest_boolq,kobest_copa,kobest_wic,kobest_hellaswag,kobest_sentineg,haerae"
TASKS_EN="mmlu,hellaswag,arc_challenge,boolq"
COMMON_ARGS="trust_remote_code=True,gpu_memory_utilization=0.70,max_model_len=2048"

# 검증할 캘리브레이션 언어. ko는 이미 확보되어 있으므로 기본값에서 제외한다.
# 여러 GPU로 병렬 실행할 때는 환경변수로 분할한다. 예:
#   CUDA_VISIBLE_DEVICES=0 CALIB_LANGS="en"    nohup bash run_calib_ablation.sh ...
#   CUDA_VISIBLE_DEVICES=3 CALIB_LANGS="mixed" nohup bash run_calib_ablation.sh ...
# 각 조합은 서로 다른 체크포인트/결과 디렉터리를 쓰므로 병렬 실행이 안전하다.
# (단, results.json은 공유되므로 효율성 지표 일부가 덮어써질 수 있다. 정확도
#  결과는 eval_* 디렉터리에 따로 저장되어 영향받지 않는다.)
CALIB_LANGS="${CALIB_LANGS:-en mixed}"
METHODS="${METHODS:-awq gptq}"

echo "############ 1단계: 재양자화 ############"
date
for LANG in ${CALIB_LANGS}; do
  for METHOD in ${METHODS}; do
    echo "=== 양자화: ${METHOD} (calib=${LANG}) ==="
    date
    python run_quant_stage.py --stage "${METHOD}" --model "${MODEL_ID}" --calib-lang "${LANG}"
    echo "--- 양자화 ${METHOD}[${LANG}] 종료 (exit=$?) ---"
    date
  done
done

echo "############ 2단계: 한국어 벤치마크 평가 ############"
date
for LANG in ${CALIB_LANGS}; do
  for METHOD in ${METHODS}; do
    echo "=== ${METHOD}[calib:${LANG}] (Korean) ==="
    date
    HF_HUB_OFFLINE=1 lm_eval --model vllm \
      --model_args "pretrained=${BASE}/${METHOD}_4bit_calib_${LANG},${COMMON_ARGS}" \
      --tasks "${TASKS_KO}" --num_fewshot 5 --batch_size auto \
      --output_path "${BASE}/eval_${METHOD}_calib_${LANG}"
    echo "--- ${METHOD}[calib:${LANG}] Korean 종료 (exit=$?) ---"
    date
  done
done

echo "############ 3단계: 영어 벤치마크 평가 ############"
date
for LANG in ${CALIB_LANGS}; do
  for METHOD in ${METHODS}; do
    echo "=== ${METHOD}[calib:${LANG}] (English) ==="
    date
    HF_HUB_OFFLINE=1 lm_eval --model vllm \
      --model_args "pretrained=${BASE}/${METHOD}_4bit_calib_${LANG},${COMMON_ARGS}" \
      --tasks "${TASKS_EN}" --num_fewshot 5 --batch_size auto \
      --output_path "${BASE}/eval_en_${METHOD}_calib_${LANG}"
    echo "--- ${METHOD}[calib:${LANG}] English 종료 (exit=$?) ---"
    date
  done
done

echo "############ 전체 완료 ############"
date
