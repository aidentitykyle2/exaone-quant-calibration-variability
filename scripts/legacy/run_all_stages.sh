#!/bin/bash
# 하나의 venv(venv-quant) 안에서 다섯 단계를 각각 별도 프로세스로 실행한다.
# venv 전환이 필요 없다 — AWQ/GPTQ/QuaRot/SpinQuant 모두 llmcompressor로
# 통일되어 있어 라이브러리 세트가 하나뿐이기 때문이다.
# 프로세스를 분리하는 이유는 라이브러리 버전 문제가 아니라, vLLM 엔진의
# GPU 메모리가 프로세스 종료 전까지 완전히 반환되지 않을 수 있어서다.
set -e

if [ -z "$VIRTUAL_ENV" ]; then
  echo "오류: venv가 활성화되어 있지 않습니다. 먼저 다음을 실행하세요:"
  echo "  source venv-quant/bin/activate"
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
MODEL="${1:-LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct}"
# EXAONE 3.5-7.8B-Instruct는 DEFAULT_REVISION(0ff6b5e..., 2024-12-11)이
# run_quant_stage.py 안에서 자동 적용된다. 다른 모델(예: Solar)을 쓸 경우
# 그 저장소의 config.json이 transformers v5 방식으로 갱신되어 있는지
# huggingface_hub.list_repo_commits()로 먼저 확인하고, 필요하면
# 아래 python 호출에 --revision <커밋해시>를 직접 추가할 것.

echo "=== [1/5] BF16 baseline ==="
python run_quant_stage.py --stage bf16 --model "$MODEL"

echo "=== [2/5] AWQ 4-bit (llmcompressor) ==="
python run_quant_stage.py --stage awq --model "$MODEL"

echo "=== [3/5] GPTQ 4-bit (llmcompressor) ==="
python run_quant_stage.py --stage gptq --model "$MODEL"

echo "=== [4/5] QuaRot-style rotation 4-bit ==="
python run_quant_stage.py --stage quarot --model "$MODEL"

echo "=== [5/5] SpinQuant-style rotation 4-bit ==="
python run_quant_stage.py --stage spinquant --model "$MODEL"

echo "모든 단계 완료. quant_experiments/$(basename "$MODEL")/results.json 확인."
echo "다음 단계: run_eval.sh 로 KMMLU/KoBEST/HAE-RAE 등 벤치마크 평가 실행."
echo "주의: haerae 등 일부 태스크명은 lm-evaluation-harness 버전별로 다를 수 있어"
echo "      run_eval.sh 실행 전 'lm_eval --tasks list' 로 존재 여부를 확인할 것."
