#!/bin/bash
# seed_job.sh — 시드 반복 실험의 작업 하나를 실행한다.
#
# 작업 형식 (콜론 구분):
#   bf16:ko                     BF16 한국어 평가 (--log_samples)
#   bf16:en                     BF16 영어 평가   (--log_samples)
#   eval:METHOD:CALIB:SEED      기존 체크포인트 재평가 (한국어+영어). seed 0용
#   lane:METHOD:CALIB:SEED      양자화 → 한국어 평가 → 영어 평가. seed 1·2용
#   analysis                    활성화·채널·레이어 오차 분석 (M1~M3)
#
# 모든 평가는 --log_samples로 문항별 예측을 저장한다(대응 부트스트랩용).
# 결과는 기존 결과와 섞이지 않도록 seed_eval/ 하위에 따로 저장한다.
#
# 7.8B 기존 실험과 조건을 맞추기 위해 gpu_memory_utilization=0.70을 유지한다.
# (0.90은 32B 전용 설정이다. 여기서 바꾸면 기존 결과와 비교가 어긋난다.)
#
# 사용법:
#   CUDA_VISIBLE_DEVICES=0 bash seed_job.sh lane:gptq:ko:1

set -u

JOB="${1:?작업을 지정해야 한다. 예: lane:gptq:ko:1}"
: "${CUDA_VISIBLE_DEVICES:?CUDA_VISIBLE_DEVICES를 지정해야 한다}"

# 모델과 저장 경로는 환경변수로 덮어쓸 수 있다(32B 등 다른 모델에 재사용).
#   MODEL_ID=LGAI-EXAONE/EXAONE-4.0-32B BASE=quant_experiments/EXAONE-4.0-32B \
#   REVISION= GPU_UTIL=0.90 bash seed_job.sh sweep:gptq:ko:1:256
# REVISION을 빈 문자열로 두면 리비전을 고정하지 않는다(공식 통합 모델은 불필요).
MODEL_ID="${MODEL_ID:-LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct}"
REVISION="${REVISION-0ff6b5ec7c13b049b253a16a889aa269e6b79a94}"
BASE="${BASE:-quant_experiments/EXAONE-3.5-7.8B-Instruct}"
OUT="${BASE}/seed_eval"
TASKS_KO="kmmlu,kobest_boolq,kobest_copa,kobest_wic,kobest_hellaswag,kobest_sentineg,haerae"
TASKS_EN="mmlu,hellaswag,arc_challenge,boolq"
# 7.8B는 0.70, 32B는 0.90(BF16 가중치만 약 64GB)
GPU_UTIL="${GPU_UTIL:-0.70}"
COMMON="trust_remote_code=True,gpu_memory_utilization=${GPU_UTIL},max_model_len=2048"

# 체크포인트 경로 규칙 (run_quant_stage.py와 동일)
ckpt_path () {
  local m="$1" c="$2" s="$3" name
  name="${m}_4bit"
  [ "${c}" != "ko" ] && name="${name}_calib_${c}"
  [ "${s}" != "0" ] && name="${name}_s${s}"
  echo "${BASE}/${name}"
}

run_eval () {  # $1=pretrained 인자, $2=ko|en, $3=출력 이름
  local args="$1" lang="$2" name="$3" tasks offline=""
  if [ "${lang}" = "ko" ]; then tasks="${TASKS_KO}"; offline="HF_HUB_OFFLINE=1"
  else tasks="${TASKS_EN}"; fi
  echo "=== 평가 [${lang}] ${name} (GPU ${CUDA_VISIBLE_DEVICES}) ==="
  date
  env ${offline} lm_eval --model vllm \
    --model_args "${args},${COMMON}" \
    --tasks "${tasks}" --num_fewshot 5 --batch_size auto \
    --log_samples \
    --output_path "${OUT}/${lang}/${name}"
  local rc=$?
  echo "--- 평가 [${lang}] ${name} 종료 (exit=${rc}) ---"
  date
  return ${rc}
}

IFS=':' read -r TYPE A1 A2 A3 A4 <<< "${JOB}"

case "${TYPE}" in
  bf16)
    REV_ARG=""
    [ -n "${REVISION}" ] && REV_ARG=",revision=${REVISION}"
    run_eval "pretrained=${MODEL_ID},dtype=bfloat16${REV_ARG}" "${A1}" "bf16"
    ;;

  eval)
    CK="$(ckpt_path "${A1}" "${A2}" "${A3}")"
    if [ ! -d "${CK}" ]; then echo "오류: 체크포인트 없음 → ${CK}"; exit 1; fi
    run_eval "pretrained=${CK}" ko "${A1}_${A2}_s${A3}"
    run_eval "pretrained=${CK}" en "${A1}_${A2}_s${A3}"
    ;;

  lane)
    echo "=== 양자화 ${A1}/${A2}/seed${A3} (GPU ${CUDA_VISIBLE_DEVICES}) ==="
    date
    REV_OPT=""
    [ -n "${REVISION}" ] && REV_OPT="--revision ${REVISION}"
    python run_quant_stage.py --stage "${A1}" --model "${MODEL_ID}" ${REV_OPT} \
      --calib-lang "${A2}" --calib-seed "${A3}"
    QRC=$?
    echo "--- 양자화 ${A1}/${A2}/seed${A3} 종료 (exit=${QRC}) ---"
    date
    if [ ${QRC} -ne 0 ]; then echo "양자화 실패로 평가를 건너뛴다."; exit 1; fi
    CK="$(ckpt_path "${A1}" "${A2}" "${A3}")"
    if [ ! -d "${CK}" ]; then echo "오류: 체크포인트 없음 → ${CK}"; exit 1; fi
    run_eval "pretrained=${CK}" ko "${A1}_${A2}_s${A3}"
    run_eval "pretrained=${CK}" en "${A1}_${A2}_s${A3}"
    ;;

  sweep)
    # sweep:METHOD:CALIB:SEED:N — 캘리브레이션 크기 실험(E2)
    # 양자화 → KMMLU → MMLU. 크기·시드 분석에 필요한 두 벤치마크만 평가해 시간을 줄인다.
    N="${A4:-256}"
    echo "=== 양자화 ${A1}/${A2}/n${N}/seed${A3} (GPU ${CUDA_VISIBLE_DEVICES}) ==="
    date
    REV_OPT=""
    [ -n "${REVISION}" ] && REV_OPT="--revision ${REVISION}"
    python run_quant_stage.py --stage "${A1}" --model "${MODEL_ID}" ${REV_OPT} \
      --calib-lang "${A2}" --calib-seed "${A3}" --calib-n "${N}"
    QRC=$?
    echo "--- 양자화 종료 (exit=${QRC}) ---"; date
    [ ${QRC} -ne 0 ] && { echo "양자화 실패로 평가를 건너뛴다."; exit 1; }
    name="${A1}_4bit"
    [ "${A2}" != "ko" ] && name="${name}_calib_${A2}"
    [ "${N}" != "256" ] && name="${name}_n${N}"
    [ "${A3}" != "0" ] && name="${name}_s${A3}"
    CK="${BASE}/${name}"
    [ -d "${CK}" ] || { echo "오류: 체크포인트 없음 → ${CK}"; exit 1; }
    OUTNAME="${A1}_${A2}_s${A3}"
    [ "${N}" != "256" ] && OUTNAME="${A1}_${A2}_n${N}_s${A3}"
    TASKS_KO="kmmlu" TASKS_EN="mmlu" run_eval "pretrained=${CK}" ko "${OUTNAME}"
    TASKS_KO="kmmlu" TASKS_EN="mmlu" run_eval "pretrained=${CK}" en "${OUTNAME}"
    ;;

  evalone)
    # evalone:체크포인트_디렉터리명:출력이름:언어
    # 기존 체크포인트를 한 언어로만 재평가한다(--log_samples). 경로 규칙이 다른
    # 체크포인트(예: quarot_4bit_rtn_backup)도 디렉터리명을 그대로 지정해 쓸 수 있고,
    # 한국어·영어를 별도 작업으로 나눠 동적 큐에서 GPU 간 균형을 맞춘다.
    CK="${BASE}/${A1}"
    if [ ! -d "${CK}" ]; then echo "오류: 체크포인트 없음 → ${CK}"; exit 1; fi
    run_eval "pretrained=${CK}" "${A3}" "${A2}"
    ;;

  analysis)
    echo "=== 활성화·채널·레이어 오차 분석 (GPU ${CUDA_VISIBLE_DEVICES}) ==="
    date
    python activation_analysis.py --base "${BASE}" --out "${BASE}/mechanism"
    echo "--- 분석 종료 (exit=$?) ---"
    date
    ;;

  *)
    echo "알 수 없는 작업 유형: ${TYPE}"; exit 1 ;;
esac
