#!/bin/bash
# launch_lanes.sh — 사용 가능한 GPU에 레인을 분배해 동시 실행
#
# 한 GPU가 여러 조건을 순차 처리해야 하는 경우(GPU 수 < 조건 수)를 위해
# 체이닝을 지원한다. "|" 로 구분하면 같은 GPU에서 순서대로 실행된다.
#
# LANES 형식:  "GPU:METHOD:CALIB[|METHOD:CALIB...]" 를 공백으로 구분
#
# 예시 — GPU 3장(0,2,3)에 조건 4개:
#   LANES="0:awq:ko 2:awq:en 3:gptq:ko|gptq:en" bash launch_lanes.sh
#     GPU 0 → AWQ[ko]                    약 17시간
#     GPU 2 → AWQ[en]                    약 17시간
#     GPU 3 → GPTQ[ko] → GPTQ[en]        약 22시간  ← 전체를 결정
#
# 예시 — GPU 4장일 때:
#   LANES="0:awq:ko 1:awq:en 2:gptq:ko 3:gptq:en" bash launch_lanes.sh
#
# 예시 — 안전 축소안(GPTQ만 + 소형 모델):
#   LANES="0:gptq:ko 2:gptq:en" bash launch_lanes.sh
#
# 사용법:
#   cd /path/to/quant_exp
#   source venv-quant/bin/activate
#   export HF_HOME=/path/to/.cache/huggingface
#   LANES="..." bash launch_lanes.sh

set -u

if [ -z "${VIRTUAL_ENV:-}" ]; then
  echo "오류: venv 미활성화. source venv-quant/bin/activate 먼저 실행할 것."
  exit 1
fi
case "${VIRTUAL_ENV}" in
  */quant_exp/venv-quant) ;;
  *) echo "경고: 예상과 다른 venv다 → ${VIRTUAL_ENV}"
     echo "      /path/to/quant_exp/venv-quant 인지 확인할 것."
     read -p "계속하겠는가? (y/N) " a
     [ "$a" = "y" ] || [ "$a" = "Y" ] || exit 1 ;;
esac

MODEL_ID="${MODEL_ID:-LGAI-EXAONE/EXAONE-4.0-32B}"
LANES="${LANES:-}"
STAGGER="${STAGGER:-180}"

if [ -z "${LANES}" ]; then
  echo "오류: LANES를 지정해야 한다."
  echo '예: LANES="0:awq:ko 2:awq:en 3:gptq:ko|gptq:en" bash launch_lanes.sh'
  exit 1
fi

echo "모델      : ${MODEL_ID}"
echo "레인 구성 : ${LANES}"
echo "시작 간격 : ${STAGGER}초"
echo ""
echo "현재 GPU 상태:"
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv
echo ""
echo "주의: 다른 사용자의 프로세스가 점유 중인 GPU는 배정하지 말 것."
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv 2>/dev/null | head -10
echo ""
read -p "위 상태로 진행하겠는가? (y/N) " ans
if [ "${ans}" != "y" ] && [ "${ans}" != "Y" ]; then
  echo "중단한다."
  exit 0
fi

first=1
for entry in ${LANES}; do
  GPU="${entry%%:*}"
  CHAIN="${entry#*:}"          # "awq:ko" 또는 "gptq:ko|gptq:en"

  if [ "${first}" -eq 0 ]; then
    echo "  (${STAGGER}초 대기)"
    sleep "${STAGGER}"
  fi
  first=0

  # 체인을 하나의 백그라운드 작업으로 묶어 순차 실행한다.
  LOG="chain_gpu${GPU}.txt"
  (
    IFS='|'
    for step in ${CHAIN}; do
      METHOD="${step%%:*}"
      CALIB="${step#*:}"
      echo "############ GPU ${GPU}: ${METHOD}/${CALIB} 시작 ############"
      date
      bash run_lane.sh "${GPU}" "${METHOD}" "${CALIB}" "${MODEL_ID}"
      echo "############ GPU ${GPU}: ${METHOD}/${CALIB} 종료 (exit=$?) ############"
      date
    done
    echo "############ GPU ${GPU} 체인 전체 완료 ############"
    date
  ) > "${LOG}" 2>&1 &
  disown

  echo "시작: GPU ${GPU} ← ${CHAIN}   (로그: ${LOG})"
done

echo ""
echo "모든 체인을 백그라운드로 시작했다."
echo ""
echo "진행 확인:"
echo "  nvidia-smi"
echo "  grep -hE '^####|^===|^---' chain_gpu*.txt"
echo "  tail -2 chain_gpu0.txt"
