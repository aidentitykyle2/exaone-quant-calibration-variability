#!/bin/bash
# launch_seed_exp.sh — ①②③ 실험을 H100 4장에 분배해 실행
#
# 작업 배치 원칙
#   - 짧은 작업(분석, BF16, seed 0 재평가)을 앞에 둔다. 약 2.5시간 안에
#     ② 기제 분석과 ③ seed 0 대응 부트스트랩이 가능한 상태가 되어,
#     긴 시드 반복(①)을 기다리지 않고도 중간 결과를 확인할 수 있다.
#   - seed 0 재평가가 기존 수치와 일치하는지로 파이프라인 정합성을 조기 검증한다.
#   - 가장 긴 작업(AWQ 양자화)을 GPU마다 하나씩 나눠 전체 시간을 균형 있게 맞춘다.
#
# 예상 소요 (7.8B 실측 기반)
#   GPU 0: 분석 60m → seed0 재평가 95m → AWQ s1 200m → GPTQ s1 135m  = 490m
#   GPU 1: BF16 ko 55m → seed0 재평가 95m → AWQ s2 200m → GPTQ s2 135m = 485m
#   GPU 2: BF16 en 40m → seed0 재평가 95m → AWQ s1 200m → GPTQ s1 135m = 470m
#   GPU 3: seed0 재평가 95m → AWQ s2 200m → GPTQ s2 135m              = 430m
#   전체 ≈ 8시간 10분
#
# 큐는 QUEUE_0 ~ QUEUE_3 환경변수로 덮어쓸 수 있다(공백 구분).
#
# 사용법:
#   cd /path/to/quant_exp
#   source venv-quant/bin/activate
#   export HF_HOME=/path/to/.cache/huggingface
#   bash launch_seed_exp.sh

set -u

if [ -z "${VIRTUAL_ENV:-}" ]; then
  echo "오류: venv 미활성화. source venv-quant/bin/activate 먼저 실행할 것."; exit 1
fi
case "${VIRTUAL_ENV}" in
  */quant_exp/venv-quant) ;;
  *) echo "경고: 예상과 다른 venv → ${VIRTUAL_ENV}"
     read -p "계속하겠는가? (y/N) " a; [ "$a" = "y" ] || [ "$a" = "Y" ] || exit 1 ;;
esac

EI="en_instruct"
QUEUE_0="${QUEUE_0:-analysis eval:gptq:ko:0 lane:awq:ko:1 lane:gptq:ko:1}"
QUEUE_1="${QUEUE_1:-bf16:ko eval:gptq:${EI}:0 lane:awq:ko:2 lane:gptq:ko:2}"
QUEUE_2="${QUEUE_2:-bf16:en eval:awq:ko:0 lane:awq:${EI}:1 lane:gptq:${EI}:1}"
QUEUE_3="${QUEUE_3:-eval:awq:${EI}:0 lane:awq:${EI}:2 lane:gptq:${EI}:2}"
STAGGER="${STAGGER:-180}"

echo "GPU 0: ${QUEUE_0}"
echo "GPU 1: ${QUEUE_1}"
echo "GPU 2: ${QUEUE_2}"
echo "GPU 3: ${QUEUE_3}"
echo ""
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv
nvidia-smi --query-compute-apps=pid,used_memory --format=csv 2>/dev/null
echo ""
echo "다른 사용자가 점유한 GPU가 있으면 중단하고 해당 큐를 조정할 것."
read -p "진행하겠는가? (y/N) " ans
[ "${ans}" = "y" ] || [ "${ans}" = "Y" ] || { echo "중단한다."; exit 0; }

mkdir -p logs_seed
first=1
for g in 0 1 2 3; do
  var="QUEUE_${g}"
  jobs="${!var}"
  [ -z "${jobs}" ] && continue
  if [ ${first} -eq 0 ]; then echo "  (${STAGGER}초 대기)"; sleep "${STAGGER}"; fi
  first=0
  (
    for job in ${jobs}; do
      echo "######## GPU ${g}: ${job} 시작 ########"; date
      CUDA_VISIBLE_DEVICES="${g}" bash seed_job.sh "${job}"
      echo "######## GPU ${g}: ${job} 종료 (exit=$?) ########"; date
    done
    echo "######## GPU ${g} 큐 전체 완료 ########"; date
  ) > "logs_seed/gpu${g}.txt" 2>&1 &
  disown
  echo "시작: GPU ${g}  (로그: logs_seed/gpu${g}.txt)"
done

echo ""
echo "진행 확인:"
echo "  grep -h '########' logs_seed/gpu*.txt"
echo "  nvidia-smi"
