#!/bin/bash
# run_32b.sh — EXAONE 4.0-32B 검증 실험 실행기
#
# 7.8B에서 쓴 도구를 환경변수로 32B에 맞춰 재사용한다.
#   MODEL_ID  대상 모델 (4.5는 비전-언어 모델이므로 쓰지 않는다)
#   REVISION  빈 문자열 = 리비전 고정 안 함 (4.0은 transformers 공식 통합)
#   GPU_UTIL  0.90 (BF16 가중치만 약 64GB)
#
# 사용법:
#   bash run_32b.sh check     사전 확인 (GPU 미사용, 약 5분)
#   bash run_32b.sh bf16      BF16 기준선 (GPU 0,1 = 한국어 TP2 / GPU 2,3 = 영어 TP2)
#   bash run_32b.sh lanes     양자화 6건 (동적 큐, 비어 있는 GPU가 가져감)
#   bash run_32b.sh ppl       PPL 측정 (양자화 완료 후)
#   bash run_32b.sh analyze   분석 일괄 실행 (CPU)

set -u
export MODEL_ID="${MODEL_ID:-LGAI-EXAONE/EXAONE-4.0-32B}"
export REVISION=""
export BASE="${BASE:-quant_experiments/EXAONE-4.0-32B}"
export GPU_UTIL="${GPU_UTIL:-0.90}"
CMD="${1:?check | bf16 | lanes | ppl | analyze}"

[ -z "${VIRTUAL_ENV:-}" ] && { echo "venv 미활성화"; exit 1; }
echo "모델 ${MODEL_ID} / 저장 ${BASE} / gpu_util ${GPU_UTIL}"

case "${CMD}" in
check)
  echo "=== 1. 설정 확인 (아키텍처·크기) ==="
  CUDA_VISIBLE_DEVICES="" python -c "
from transformers import AutoConfig
c = AutoConfig.from_pretrained('${MODEL_ID}')
print(' architectures :', c.architectures)
print(' layers        :', c.num_hidden_layers)
print(' hidden_size   :', c.hidden_size)
print(' heads / kv    :', c.num_attention_heads, '/', c.num_key_value_heads)
print(' intermediate  :', c.intermediate_size, '(2의 거듭제곱이 아니면 QuaRot R4 제외 근거가 7.8B와 동일)')
"
  echo "=== 2. 캘리브레이션 로딩 (시드별로 다른 표본인지) ==="
  CUDA_VISIBLE_DEVICES="" python -c "
from run_quant_stage import load_calib_dataset
for l in ['ko','en_instruct']:
    print(' ', l, load_calib_dataset(l,8,seed=1)['text'] != load_calib_dataset(l,8,seed=2)['text'])
"
  echo "=== 3. GPU 상태 ==="
  nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv
  ;;
bf16)
  mkdir -p logs_32b
  KO_GPUS=0,1 EN_GPUS=2,3 nohup bash run_bf16_phase.sh > logs_32b/bf16.txt 2>&1 &
  disown
  echo "BF16 시작 (로그 logs_32b/bf16.txt). 약 4시간."
  echo "단일 GPU로 되는지 먼저 보려면: MODE=single bash run_bf16_phase.sh"
  ;;
lanes)
  mkdir -p logs_32b
  for g in 0 1 2 3; do
    nohup bash queue_worker.sh $g e32_jobs.txt > logs_32b/gpu$g.txt 2>&1 &
    disown
    echo "  워커 시작: GPU $g"
    sleep 180
  done
  echo "6건을 4장이 나눠 처리한다. 예상 약 20~22시간."
  ;;
ppl)
  mkdir -p logs_32b
  CUDA_VISIBLE_DEVICES=0 python ppl_eval.py measure --base "${BASE}" --bf16-only
  CUDA_VISIBLE_DEVICES=0 nohup python ppl_eval.py measure --base "${BASE}" --conds ko          --seeds 1 2 3 > logs_32b/ppl0.txt 2>&1 & disown
  CUDA_VISIBLE_DEVICES=1 nohup python ppl_eval.py measure --base "${BASE}" --conds en_instruct --seeds 1 2 3 > logs_32b/ppl1.txt 2>&1 & disown
  echo "PPL 측정 시작. 약 30분."
  ;;
analyze)
  python paired_bootstrap.py --base "${BASE}" --scoring tie_split
  python single_run_risk.py  --base "${BASE}" --seeds random
  python m4_full.py          --base "${BASE}"
  python final_analysis.py   --base "${BASE}"
  python ppl_eval.py aggregate --base "${BASE}"
  python export_calib_manifests.py --base "${BASE}" --jobs e32_jobs.txt
  ;;
*) echo "알 수 없는 명령: ${CMD}"; exit 1 ;;
esac
