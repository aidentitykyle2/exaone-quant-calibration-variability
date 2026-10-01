#!/bin/bash
# run_family.sh — 다른 모델 계열로 같은 설계를 반복 실행 (향후 연구용)
#
# *** 본 논문의 결과에는 사용되지 않았다. ***
#
# 기존 인프라(seed_job.sh / queue_worker.sh)를 그대로 쓴다. seed_job.sh는
# MODEL_ID / REVISION / BASE / GPU_UTIL 을 환경변수로 받도록 이미 만들어 두었으므로
# 스크립트를 고칠 필요가 없다 — 이것이 본 논문 코드를 손대지 않고 확장하는 경로다.
#
# 사용법:
#   source venv-quant/bin/activate
#   MID=Qwen/Qwen2.5-7B-Instruct nohup bash run_family.sh > run_family.txt 2>&1 & disown

set -u
MID="${MID:-Qwen/Qwen2.5-7B-Instruct}"
REV="${REV:-}"                              # 공식 지원 모델은 비워 둔다
BASE="${BASE:-quant_experiments/$(basename "${MID}")}"
GPUS="${GPUS:-0 1 2 3}"
GPU_UTIL="${GPU_UTIL:-0.70}"
LOGDIR="${LOGDIR:-logs_family/$(basename "${MID}")}"
JOBS="${JOBS:-jobs_family.txt}"

[ -z "${VIRTUAL_ENV:-}" ] && { echo "오류: venv 미활성화."; exit 1; }
[ -f "${JOBS}" ] || { echo "오류: ${JOBS} 가 없다."; exit 1; }
mkdir -p "${LOGDIR}" "${BASE}"

echo "######## 모델 계열 확장 실험 ########"
echo "모델   : ${MID}  (리비전 ${REV:-main})"
echo "저장   : ${BASE}"
echo "작업   : ${JOBS}   GPU: ${GPUS}"
date
echo ""
echo "사전 점검을 통과했는지 확인한다(측정 가능성 관문 포함)."
if [ ! -f "preflight_family/$(basename "${MID}")/bf16_gate" ] && \
   [ ! -d "preflight_family/$(basename "${MID}")/bf16_gate" ]; then
  echo "경고: BF16 관문 결과가 없다. KMMLU가 우연 수준이면 20 GPU시간이 낭비된다."
  echo "      먼저 실행할 것: MID=${MID} bash preflight_family.sh gate"
  echo "      그래도 진행하려면 FORCE=1 을 지정할 것."
  [ "${FORCE:-0}" != "1" ] && exit 1
fi

# 같은 작업 파일을 여러 모델에 재사용하므로 선점 기록을 모델별로 분리한다.
export QUEUE_CLAIM_DIR="queue_claims/family_$(basename "${MID}")"
mkdir -p "${QUEUE_CLAIM_DIR}"

for g in ${GPUS}; do
  MODEL_ID="${MID}" REVISION="${REV}" BASE="${BASE}" GPU_UTIL="${GPU_UTIL}" \
    nohup bash queue_worker.sh "${g}" "${JOBS}" > "${LOGDIR}/gpu${g}.txt" 2>&1 &
  echo "  GPU ${g} 워커 시작 (로그 ${LOGDIR}/gpu${g}.txt)"
  sleep 5
done
wait
echo "######## 종료 ########"; date
echo ""
echo "다음 단계:"
echo "  python rescore_all.py --base ${BASE}      # tie_split 재채점 (동일 채점 규칙)"
echo "  python final_analysis.py --base ${BASE}   # SD(X), 부호 반전 수, 두 단계 부트스트랩"
echo "  계열 간 비교는 SD(X)와 SD_calib/SE_eval 비(EXAONE 7.8B: 3.64)를 나란히 놓는다."
