#!/bin/bash
# robust_job.sh — 강건 Hessian 집계(gptq_robust) 작업 하나를 실행한다.
#
# *** 향후 연구용이다. 본 논문의 결과에는 사용되지 않았다. ***
#
# seed_job.sh를 고치지 않고 따로 둔 이유: seed_job.sh는 본 논문 결과를 만든
# 스크립트이므로 손대면 재현 기록이 흐려진다. 평가 태스크·GPU 설정·경로 규칙은
# seed_job.sh와 동일하게 맞췄다(비교 가능성이 이 실험의 전부다).
#
# 작업 형식:
#   rcheck:CALIB:SEED:K            배선 검증. agg=mean으로 양자화한 뒤, 같은 조건의
#                                  기준선 gptq 체크포인트와 가중치를 직접 비교한다.
#                                  mean은 기존 GPTQ와 수학적으로 동일하므로 일치해야 한다.
#                                  ** 이 작업이 통과하기 전에는 아래 rlane을 돌리지 말 것 **
#   rlane:CALIB:SEED:K:AGG         양자화 → KMMLU → MMLU 평가 (본 실험)
#
# 사용법:
#   CUDA_VISIBLE_DEVICES=0 bash robust_job.sh rcheck:ko:1:4
#   nohup bash queue_worker.sh 0 jobs_robust.txt > logs_robust/gpu0.txt 2>&1 & disown
#     ※ queue_worker.sh는 seed_job.sh를 호출하므로, 강건 실험에는
#        QUEUE_JOB_SCRIPT=robust_job.sh 를 지원하는 queue_worker.sh가 필요하다.
#        패치는 robust_patch.md의 "queue_worker.sh 한 줄 변경" 절을 볼 것.

set -u

JOB="${1:?작업을 지정해야 한다. 예: rlane:ko:1:4:geomedian}"
: "${CUDA_VISIBLE_DEVICES:?CUDA_VISIBLE_DEVICES를 지정해야 한다}"

MODEL_ID="${MODEL_ID:-LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct}"
REVISION="${REVISION-0ff6b5ec7c13b049b253a16a889aa269e6b79a94}"
BASE="${BASE:-quant_experiments/EXAONE-3.5-7.8B-Instruct}"
OUT="${BASE}/seed_eval"
GPU_UTIL="${GPU_UTIL:-0.70}"
CALIB_N="${CALIB_N:-256}"
COMMON="trust_remote_code=True,gpu_memory_utilization=${GPU_UTIL},max_model_len=2048"

REV_OPT=""
[ -n "${REVISION}" ] && REV_OPT="--revision ${REVISION}"

# gptq_robust 체크포인트 경로 규칙 (run_quant_stage.py 패치와 동일)
#   gptq_robust_4bit[_calib_<lang>][_n<N>][_s<seed>]_k<K>_<agg>
robust_ckpt () {
  local c="$1" s="$2" k="$3" agg="$4" name="gptq_robust_4bit"
  [ "${c}" != "ko" ] && name="${name}_calib_${c}"
  [ "${CALIB_N}" != "256" ] && name="${name}_n${CALIB_N}"
  [ "${s}" != "0" ] && name="${name}_s${s}"
  echo "${BASE}/${name}_k${k}_${agg}"
}

# 기준선 gptq 체크포인트 경로 (seed_job.sh와 동일)
base_ckpt () {
  local c="$1" s="$2" name="gptq_4bit"
  [ "${c}" != "ko" ] && name="${name}_calib_${c}"
  [ "${CALIB_N}" != "256" ] && name="${name}_n${CALIB_N}"
  [ "${s}" != "0" ] && name="${name}_s${s}"
  echo "${BASE}/${name}"
}

run_eval () {  # $1=pretrained 인자, $2=ko|en, $3=출력 이름
  local args="$1" lang="$2" name="$3" tasks offline=""
  if [ "${lang}" = "ko" ]; then tasks="kmmlu"; offline="HF_HUB_OFFLINE=1"
  else tasks="mmlu"; fi
  echo "=== 평가 [${lang}] ${name} (GPU ${CUDA_VISIBLE_DEVICES}) ==="; date
  env ${offline} lm_eval --model vllm \
    --model_args "${args},${COMMON}" \
    --tasks "${tasks}" --num_fewshot 5 --batch_size auto \
    --log_samples \
    --output_path "${OUT}/${lang}/${name}"
  local rc=$?
  echo "--- 평가 [${lang}] ${name} 종료 (exit=${rc}) ---"; date
  return ${rc}
}

IFS=':' read -r TYPE A1 A2 A3 A4 <<< "${JOB}"

case "${TYPE}" in
  rcheck)
    # A1=calib, A2=seed, A3=K.  agg는 mean으로 고정한다(대조군의 정의).
    CALIB="${A1}"; SEED="${A2}"; K="${A3}"
    BASE_CK="$(base_ckpt "${CALIB}" "${SEED}")"
    if [ ! -d "${BASE_CK}" ]; then
      echo "오류: 비교 기준이 되는 기준선 체크포인트가 없다 → ${BASE_CK}"
      echo "      본 논문 실험에서 만든 gptq 체크포인트를 먼저 확보할 것."
      exit 1
    fi
    echo "=== 배선 검증: gptq_robust(K=${K}, agg=mean) vs 기준선 gptq ==="; date
    python run_quant_stage.py --stage gptq_robust --model "${MODEL_ID}" ${REV_OPT} \
      --calib-lang "${CALIB}" --calib-seed "${SEED}" --calib-n "${CALIB_N}" \
      --robust-k "${K}" --robust-agg mean
    QRC=$?
    echo "--- 양자화 종료 (exit=${QRC}) ---"; date
    [ ${QRC} -ne 0 ] && { echo "양자화 실패."; exit 1; }
    ROB_CK="$(robust_ckpt "${CALIB}" "${SEED}" "${K}" mean)"
    [ -d "${ROB_CK}" ] || { echo "오류: 체크포인트 없음 → ${ROB_CK}"; exit 1; }
    python compare_ckpt.py "${BASE_CK}" "${ROB_CK}" \
      --report "${BASE}/robust_check_${CALIB}_s${SEED}_k${K}.json"
    CRC=$?
    echo "--- 가중치 비교 종료 (exit=${CRC}) ---"
    if [ ${CRC} -ne 0 ]; then
      echo "배선 검증 실패. 집계 경로에 기준선과 다른 요인이 섞여 있다."
      echo "가장 흔한 원인 두 가지:"
      echo "  1) DISABLE_QAC_MODIFIERS에 RobustGPTQModifier가 등록되지 않았다"
      echo "     (robust_gptq.py 하단에서 처리하므로, import가 실제로 되었는지 확인)"
      echo "  2) 부분집합 배정이 배치 경계와 어긋나 표본이 중복·누락되었다"
      echo "     (로그의 '부분집합 표본 수' 합이 ${CALIB_N}인지 확인)"
      exit 1
    fi
    echo "배선 검증 통과. 본 실험(rlane)을 진행해도 된다."
    ;;

  rlane)
    # A1=calib, A2=seed, A3=K, A4=agg
    CALIB="${A1}"; SEED="${A2}"; K="${A3}"; AGG="${A4:-geomedian}"
    echo "=== 양자화 gptq_robust/${CALIB}/seed${SEED}/K${K}/${AGG} (GPU ${CUDA_VISIBLE_DEVICES}) ==="
    date
    python run_quant_stage.py --stage gptq_robust --model "${MODEL_ID}" ${REV_OPT} \
      --calib-lang "${CALIB}" --calib-seed "${SEED}" --calib-n "${CALIB_N}" \
      --robust-k "${K}" --robust-agg "${AGG}"
    QRC=$?
    echo "--- 양자화 종료 (exit=${QRC}) ---"; date
    [ ${QRC} -ne 0 ] && { echo "양자화 실패로 평가를 건너뛴다."; exit 1; }
    CK="$(robust_ckpt "${CALIB}" "${SEED}" "${K}" "${AGG}")"
    [ -d "${CK}" ] || { echo "오류: 체크포인트 없음 → ${CK}"; exit 1; }
    NAME="gptq_robust_k${K}_${AGG}_${CALIB}_s${SEED}"
    run_eval "pretrained=${CK}" ko "${NAME}"
    run_eval "pretrained=${CK}" en "${NAME}"
    ;;

  *)
    echo "알 수 없는 작업 유형: ${TYPE} (rcheck 또는 rlane)"; exit 1 ;;
esac
