#!/bin/bash
# run_robust.sh — 강건 Hessian 집계 실험 실행 (향후 연구용)
#
# *** 본 논문의 결과에는 사용되지 않았다. ***
#
# 두 단계로 나뉘고, 1단계가 실패하면 2단계로 넘어가지 않는다.
#   1단계 배선 검증 : agg=mean이 기준선 GPTQ와 같은 가중치를 내는지 (GPU 1장, 순차)
#   2단계 본 실험   : agg=geomedian, 18시드 x 2언어 (GPU 4장, 동적 큐)
#
# 순서를 강제하는 이유: 배선 검증 없이 본 실험을 돌리면, SD(X)가 줄든 늘든
# 그것이 집계 방식 때문인지 구현 오류 때문인지 구분할 수 없다. 본 논문이
# 단일 실행 결과를 믿지 말라고 주장하는 만큼, 완화 절차에도 같은 기준을 적용한다.
#
# 사용법:
#   source venv-quant/bin/activate
#   nohup bash run_robust.sh > run_robust.txt 2>&1 & disown
#
# 다른 모델에 쓰려면 환경변수를 덮어쓴다(robust_job.sh와 동일한 인터페이스):
#   MODEL_ID=... REVISION=... BASE=... GPU_UTIL=0.90 bash run_robust.sh

set -u

GPUS="${GPUS:-0 1 2 3}"
CHECK_GPU="${CHECK_GPU:-0}"
LOGDIR="${LOGDIR:-logs_robust}"
SKIP_CHECK="${SKIP_CHECK:-0}"

[ -z "${VIRTUAL_ENV:-}" ] && { echo "오류: venv 미활성화."; exit 1; }
for f in robust_gptq.py robust_job.sh compare_ckpt.py jobs_robust.txt jobs_robust_check.txt; do
  [ -f "${f}" ] || { echo "오류: ${f} 가 없다."; exit 1; }
done
grep -q 'gptq_robust' run_quant_stage.py || {
  echo "오류: run_quant_stage.py에 gptq_robust 단계가 없다."
  echo "      patch -p0 run_quant_stage.py < robust_gptq.patch 를 먼저 적용할 것."
  exit 1
}
grep -q 'QUEUE_JOB_SCRIPT' queue_worker.sh || {
  echo "오류: queue_worker.sh가 QUEUE_JOB_SCRIPT를 지원하지 않는다."
  echo "      patch -p0 queue_worker.sh < queue_worker.patch 를 먼저 적용할 것."
  exit 1
}
mkdir -p "${LOGDIR}"

echo "######## 0단계: 집계 수식 자체 점검 (GPU 불필요) ########"; date
python robust_selftest.py || { echo "집계 수식 점검 실패. 여기서 멈춘다."; exit 1; }
echo ""

if [ "${SKIP_CHECK}" != "1" ]; then
  echo "######## 1단계: 배선 검증 (GPU ${CHECK_GPU}, 순차) ########"; date
  QUEUE_JOB_SCRIPT=robust_job.sh \
    bash queue_worker.sh "${CHECK_GPU}" jobs_robust_check.txt \
    2>&1 | tee "${LOGDIR}/check.txt"
  # queue_worker는 개별 작업의 종료코드를 선점 디렉터리에 남긴다.
  bad=0
  for f in queue_claims/jobs_robust_check/*/exit_code; do
    [ -f "${f}" ] || continue
    [ "$(cat "${f}")" = "0" ] || { echo "실패한 검증 작업: ${f}"; bad=1; }
  done
  if [ "${bad}" != "0" ]; then
    echo ""
    echo "배선 검증 실패. 본 실험을 돌리지 않는다."
    echo "로그: ${LOGDIR}/check.txt 와 quant_experiments/*/robust_check_*.json 확인."
    exit 1
  fi
  echo "배선 검증 통과."
  echo ""
fi

echo "######## 2단계: 본 실험 (GPU ${GPUS}, 동적 큐) ########"; date
for g in ${GPUS}; do
  QUEUE_JOB_SCRIPT=robust_job.sh \
    nohup bash queue_worker.sh "${g}" jobs_robust.txt \
    > "${LOGDIR}/gpu${g}.txt" 2>&1 &
  echo "  GPU ${g} 워커 시작 (로그 ${LOGDIR}/gpu${g}.txt)"
  sleep 5      # vLLM·HF 캐시 초기화가 겹치지 않도록 약간 띄운다
done
wait
echo "######## 본 실험 종료 ########"; date
echo ""
echo "다음 단계:"
echo "  python rescore_all.py    # tie_split 재채점 (기준선과 같은 채점 규칙을 써야 한다)"
echo "  python final_analysis.py # SD(X)를 기준선 2.00 %p 와 비교"
