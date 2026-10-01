#!/bin/bash
# run_72h.sh — H100 4장 72시간 계획 (2트랙 병렬)
#
# 트랙을 모델별로 나누는 이유
#   32B와 7.8B는 MODEL_ID·GPU_UTIL이 달라 하나의 동적 큐로 섞을 수 없다.
#   GPU를 반씩 전담시키면 두 트랙의 총 시간이 비슷해 균형이 맞는다.
#
#   GPU 0,1 → EXAONE 3.5-32B  : BF16 → GPTQ 5시드 × 2조건   약 52시간
#   GPU 2,3 → EXAONE 3.5-7.8B : GPTQ 시드 11~17, AWQ 3~5, n=512  약 48시간
#
# 32B를 3.5 세대로 쓰는 이유
#   7.8B와 같은 세대라 규모만 변수가 된다. 4.0-32B는 아키텍처 세대까지 바뀌어
#   "규모 효과"라는 주장을 세울 수 없다. 검증된 AWQ·QuaRot 매핑도 그대로 쓸 수 있다.
#
# 우선순위 근거 (7.8B 현재 결과 기준)
#   최우선 32B      "이 모델에서만 나타나는 현상 아닌가"에 답할 수 있는 유일한 실험
#   최우선 시드확대  X 분포(평균이 아니라 분포)를 제시하려면 17시드 이상이 필요
#   상위   AWQ      GPTQ 구현 특성인지 PTQ 전반의 문제인지 구별
#   상위   n=512    128·256·1024가 평탄했으므로 곡선의 빈 구간을 메운다
#   후순위 consensus, M1/M2 시드 반복 — 결론을 바꾸지 않으므로 여유 시간에만
#
# 사용법:
#   nohup bash run_72h.sh > run72.txt 2>&1 & disown
#   TRACK=32b bash run_72h.sh    # 한쪽만
#   TRACK=78b bash run_72h.sh

set -u
M32="${M32:-LGAI-EXAONE/EXAONE-3.5-32B-Instruct}"
M78="${M78:-LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct}"
R78="${R78:-0ff6b5ec7c13b049b253a16a889aa269e6b79a94}"
# 32B 리비전을 반드시 고정한다.
# main(3d27aab...)의 원격 코드는 transformers 모듈식 생성판으로 갱신되어
# RopeParameters를 import하는데, 4.57.6에는 그 심볼이 없어 로드가 실패한다.
# 아래 리비전은 갱신 이전 판으로, 4.57.6에서 정상 로드됨을 확인했다.
R32="${R32:-d6fa88cd8d2c9512b40578bdc44e64909e5a5042}"
TRACK="${TRACK:-both}"
STAGGER="${STAGGER:-180}"   # 워커 시작 간격(초). 동시에 가중치를 읽는 경합을 피한다.
[ -z "${VIRTUAL_ENV:-}" ] && { echo "venv 미활성화"; exit 1; }
mkdir -p logs_72h

step () { echo ""; echo "=========================================================="; echo "# $1"; date; }

track_32b () {
  export MODEL_ID="${M32}" REVISION="${R32}" GPU_UTIL=0.90
  export BASE="quant_experiments/$(basename "${M32}")"

  step "32B 사전 확인 (설정만, GPU 미사용)"
  if [ -z "${R32}" ]; then
    echo "경고: R32(32B 리비전)가 비어 있다. main의 원격 코드가 현재 transformers와"
    echo "      호환되지 않을 수 있다. find_revision.py 로 리비전을 먼저 확인할 것."
  fi
  CUDA_VISIBLE_DEVICES="" python -c "
from transformers import AutoConfig
rev = '${R32}' or None
c = AutoConfig.from_pretrained('${M32}', revision=rev, trust_remote_code=True)
print(' layers', c.num_hidden_layers, '| hidden', c.hidden_size,
      '| heads', c.num_attention_heads, '/', c.num_key_value_heads,
      '| intermediate', c.intermediate_size)
" || { echo "32B 설정 로드 실패"; return 1; }

  step "32B BF16 기준선 (GPU 0 = 한국어 / GPU 1 = 영어, 각 단독 0.90)"
  MODE=single KO_GPUS=0 EN_GPUS=1 TASKS_KO="kmmlu" TASKS_EN="mmlu" \
    bash run_bf16_phase.sh > logs_72h/32b_bf16.txt 2>&1
  if [ ! -d "${BASE}/seed_eval/ko/bf16" ]; then
    echo "32B BF16 실패. MODE=tp2로 재시도하거나 max_model_len을 낮출 것."
    echo "  (추정 가중치 약 60 GiB, 0.90에서 KV 여유 약 24 GiB)"
    return 1
  fi

  step "32B 양자화 10건 (GPU 0,1)"
  local P=""
  for g in 0 1; do
    QUEUE_CLAIM_DIR="queue_claims/jobs_32b" bash queue_worker.sh "$g" jobs_32b.txt \
      > "logs_72h/32b_gpu${g}.txt" 2>&1 &
    P="${P} $!"; sleep "${STAGGER}"
  done
  for p in ${P}; do wait "${p}"; done

  step "32B 분석"
  {
    python paired_bootstrap.py --base "${BASE}" --scoring tie_split
    python single_run_risk.py  --base "${BASE}" --seeds random
    python m4_full.py          --base "${BASE}"
    python final_analysis.py   --base "${BASE}"
    python export_calib_manifests.py --base "${BASE}" --jobs jobs_32b.txt
  } > logs_72h/32b_analysis.txt 2>&1
  step "32B 트랙 완료"
}

track_78b () {
  export MODEL_ID="${M78}" REVISION="${R78}" GPU_UTIL=0.70
  export BASE="quant_experiments/$(basename "${M78}")"

  step "7.8B 추가 실험 30건 (GPU 2,3)"
  local P=""
  for g in 2 3; do
    QUEUE_CLAIM_DIR="queue_claims/jobs_78b" bash queue_worker.sh "$g" jobs_78b.txt \
      > "logs_72h/78b_gpu${g}.txt" 2>&1 &
    P="${P} $!"; sleep "${STAGGER}"
  done
  for p in ${P}; do wait "${p}"; done

  step "7.8B 분석 (시드 확대분 반영)"
  {
    python paired_bootstrap.py --base "${BASE}" --scoring tie_split
    python single_run_risk.py  --base "${BASE}" --seeds random
    python single_run_risk.py  --base "${BASE}" --seeds random --method awq
    python m4_full.py          --base "${BASE}"
    python final_analysis.py   --base "${BASE}"
    python consensus_select.py --base "${BASE}" --k 3
    python ppl_eval.py aggregate --base "${BASE}"
    python export_calib_manifests.py --base "${BASE}" --jobs jobs_78b.txt
  } > logs_72h/78b_analysis.txt 2>&1
  step "7.8B 트랙 완료"
}

step "72시간 계획 시작"
echo "  GPU 0,1 → ${M32}"
echo "  GPU 2,3 → ${M78}"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv

case "${TRACK}" in
  32b)  track_32b ;;
  78b)  track_78b ;;
  both) track_32b & A=$!; track_78b & B=$!; wait $A; wait $B ;;
esac

step "전체 완료"
echo "결과: quant_experiments/EXAONE-3.5-{7.8B,32B}-Instruct/{bootstrap,risk,mechanism,final,ppl}"
