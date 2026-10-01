#!/bin/bash
# queue_worker.sh — 공유 작업 목록에서 아직 아무도 가져가지 않은 작업을 하나씩 실행
#
# 여러 GPU에서 같은 작업 파일로 워커를 띄우면, 각 워커가 작업을 원자적으로
# 선점(mkdir)한 뒤 실행한다. 워커를 서로 다른 시점에 시작해도 된다.
# 지금 비어 있는 GPU로 먼저 시작하고, 다른 GPU가 비면 같은 명령으로 합류시킨다.
#
# 사용법:
#   nohup bash queue_worker.sh <GPU번호> <작업파일> > logs_rescore/gpu<N>.txt 2>&1 & disown
#
# 선점 기록은 queue_claims/<작업파일명>/ 에 남는다. 같은 작업을 다시 돌리려면
# 해당 선점 디렉터리를 지우면 된다.

set -u
GPU="${1:?GPU 번호를 지정해야 한다}"
JOBS="${2:?작업 파일을 지정해야 한다}"

if [ -z "${VIRTUAL_ENV:-}" ]; then
  echo "오류: venv 미활성화. source venv-quant/bin/activate 먼저 실행할 것."; exit 1
fi
[ -f "${JOBS}" ] || { echo "오류: 작업 파일 없음 → ${JOBS}"; exit 1; }

# 같은 작업 파일을 여러 대상(예: 1.2B와 32B)에 재사용할 때 선점 기록을 분리한다.
CLAIM_DIR="${QUEUE_CLAIM_DIR:-queue_claims/$(basename "${JOBS}" .txt)}"
mkdir -p "${CLAIM_DIR}"

echo "######## 워커 시작: GPU ${GPU}, 작업 파일 ${JOBS} ########"; date
ran=0
while IFS= read -r job || [ -n "${job}" ]; do
  job="${job%%#*}"                       # 주석 제거
  job="$(echo "${job}" | xargs)"          # 공백 제거
  [ -z "${job}" ] && continue
  key="$(echo "${job}" | tr ':/' '__')"
  if mkdir "${CLAIM_DIR}/${key}" 2>/dev/null; then
    echo "######## GPU ${GPU}: ${job} 시작 ########"; date
    CUDA_VISIBLE_DEVICES="${GPU}" bash seed_job.sh "${job}"
    rc=$?
    echo "${rc}" > "${CLAIM_DIR}/${key}/exit_code"
    echo "######## GPU ${GPU}: ${job} 종료 (exit=${rc}) ########"; date
    ran=$((ran + 1))
  fi
done < "${JOBS}"
echo "######## 워커 종료: GPU ${GPU}, 실행한 작업 ${ran}개 ########"; date
