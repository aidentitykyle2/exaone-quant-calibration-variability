# 실행 매뉴얼 (최종 확정 절차만 — 시행착오 제외)

이 문서는 지금까지의 모든 시행착오를 걷어내고, **실제로 성공이 확인된 절차만** 순서대로 정리한 것이다. 이 문서에 없는 단계는 시도할 필요가 없다.

---

## 0. 환경은 지우지 않는다

`/path/to/quant_exp/venv-quant`는 이미 검증된 환경이다 — bf16/awq/gptq/quarot 네 기법 양자화 전부 성공, bf16 정확도 평가 완주까지 확인됐다. **재설치 대상이 아니다.**

## 1. 셸 진입 시 항상 먼저 할 것

```bash
cd /path/to/quant_exp
source venv-quant/bin/activate
export CUDA_VISIBLE_DEVICES=0
export HF_HOME=/path/to/.cache/huggingface
nvidia-smi   # 다른 프로세스가 GPU를 점유하고 있지 않은지 확인
```

**주의**: 서버에 다른 프로젝트(다른 프로젝트 디렉터리)의 `venv-quant`도 존재한다. 셸 히스토리에서 잘못 골라 활성화하지 않도록 프롬프트의 경로를 항상 확인할 것.

## 2. 검증된 라이브러리 조합 (재설치할 일이 생기면 이 버전 그대로)

```
torch==2.10.0+cu126, torchvision==0.25.0, torchaudio==2.10.0
vllm==0.19.0
llmcompressor==0.10.0.3
compressed-tensors==0.14.0.1
transformers==4.57.6
```

설치 순서(한 번에 여러 패키지 설치 금지):
```bash
uv pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 \
  --index-url https://download.pytorch.org/whl/cu126
uv pip freeze | grep -E '^(torch|torchvision|torchaudio)==' > torch-constraints.txt
uv pip install -c torch-constraints.txt vllm==0.19.0 llmcompressor==0.10.0.3 \
  compressed-tensors==0.14.0.1 transformers==4.57.6 datasets numpy lm-eval
```

## 3. 양자화 실행 (이미 완료됨 — 재실행 불필요)

```bash
python run_quant_stage.py --stage bf16 --model LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct
python run_quant_stage.py --stage awq --model LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct
python run_quant_stage.py --stage gptq --model LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct
python run_quant_stage.py --stage quarot --model LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct
```

**핵심 확정 사항**:
- 모델 리비전: `0ff6b5ec7c13b049b253a16a889aa269e6b79a94`(2024-12-11, transformers v5 rope_parameters 갱신 이전) — `DEFAULT_REVISION`으로 자동 적용됨
- AWQ: EXAONE 전용 매핑 사용, GQA로 인한 `v_proj→out_proj` 제외
- QuaRot: R1+R2만 사용(R3은 AttentionInterface 미지원, R4는 Hadamard 크기 제약 + W4A16에서 수학적 항등연산이라 제외)
- SpinQuant(`learnable=True`)는 llmcompressor==0.10.0.3에 미구현 — 이번 실험 범위 밖

## 4. 정확도 평가 — 지금 여기서 막혀 있음

**완료**: BF16 (`eval_bf16/`, KMMLU 0.4555 등 확보됨)
**미완료**: AWQ, GPTQ, QuaRot

### 재개 명령 (각각 개별 실행, 하나씩 완료 확인 후 다음으로)

```bash
lm_eval --model vllm \
  --model_args pretrained=quant_experiments/EXAONE-3.5-7.8B-Instruct/awq_4bit,trust_remote_code=True,gpu_memory_utilization=0.70,max_model_len=2048 \
  --tasks kmmlu,kobest_boolq,kobest_copa,kobest_wic,kobest_hellaswag,kobest_sentineg,haerae --num_fewshot 5 --batch_size auto \
  --output_path quant_experiments/EXAONE-3.5-7.8B-Instruct/eval_awq
```
(위 명령에서 KMMLU 404가 나면 `HF_HUB_OFFLINE=1`을 명령 앞에 붙여 재시도 — 3.1절 원인 참고)

GPTQ, QuaRot도 `awq_4bit` → `gptq_4bit` / `quarot_4bit`, `eval_awq` → `eval_gptq` / `eval_quarot`로만 바꿔 동일하게 실행.

### 안전하게 돌리는 법 (터미널 연결 끊김 대비)

```bash
nohup bash -c '
lm_eval --model vllm --model_args pretrained=quant_experiments/EXAONE-3.5-7.8B-Instruct/awq_4bit,trust_remote_code=True,gpu_memory_utilization=0.70,max_model_len=2048 --tasks kmmlu,kobest_boolq,kobest_copa,kobest_wic,kobest_hellaswag,kobest_sentineg,haerae --num_fewshot 5 --batch_size auto --output_path quant_experiments/EXAONE-3.5-7.8B-Instruct/eval_awq
lm_eval --model vllm --model_args pretrained=quant_experiments/EXAONE-3.5-7.8B-Instruct/gptq_4bit,trust_remote_code=True,gpu_memory_utilization=0.70,max_model_len=2048 --tasks kmmlu,kobest_boolq,kobest_copa,kobest_wic,kobest_hellaswag,kobest_sentineg,haerae --num_fewshot 5 --batch_size auto --output_path quant_experiments/EXAONE-3.5-7.8B-Instruct/eval_gptq
lm_eval --model vllm --model_args pretrained=quant_experiments/EXAONE-3.5-7.8B-Instruct/quarot_4bit,trust_remote_code=True,gpu_memory_utilization=0.70,max_model_len=2048 --tasks kmmlu,kobest_boolq,kobest_copa,kobest_wic,kobest_hellaswag,kobest_sentineg,haerae --num_fewshot 5 --batch_size auto --output_path quant_experiments/EXAONE-3.5-7.8B-Instruct/eval_quarot
' > eval_log2.txt 2>&1 &
disown
```
진행 확인: `tail -f eval_log2.txt`

**주의**: 절대 Ctrl+C로 중단하지 말 것 — 데이터셋 다운로드 도중 끊기면 캐시가 손상되어 다음 시도에서 원인 불명 오류(이번에 겪은 KMMLU 404 등)로 이어질 수 있다. 정말 중단해야 하면 `kill`로 프로세스를 완전히 끝내고, 재시도 전 관련 캐시를 지우는 것을 권장.

## 5. 결과 취합 (모든 평가 완료 후)

```bash
python aggregate_results.py --output-dir quant_experiments/EXAONE-3.5-7.8B-Instruct
```

`results_summary.csv`(원점수), `results_degradation.csv`(BF16 대비 저하율) 확인.

## 6. 지금까지 확정된 결과 (재확인용)

| 기법 | model_loading (GiB) | kv_cache_tokens | max_concurrency | 처리량 (tok/s) |
|---|---|---|---|---|
| BF16 | 14.57 | 398,016 | 194.34x | ~1270 |
| AWQ | 4.93 | 477,248 | 233.03x | ~1650 |
| QuaRot | 4.93 | 477,728 | 233.27x | ~1080 |
| GPTQ | 4.93 | 477,728 | 233.27x | ~1690 |

| 태스크 | BF16 정확도 |
|---|---|
| KMMLU | 0.4555 |
| KoBEST BoolQ | 0.9053 |
| KoBEST COPA | 0.848 |
| KoBEST WiC | 0.677 |
| KoBEST HellaSwag | 0.59 |
| KoBEST SentiNeg | 0.9824 |
| HAE-RAE | 0.7663 |

## 7. 무해하다고 이미 확인된 경고 (무시할 것)

- `The tokenizer you are loading ... incorrect regex pattern ... fix_mistral_regex=True` — tokenizer_check로 8개 문장 전수 일치 확인됨, 실제 영향 없음
- `pretrained=... appears to be an instruct or chat variant but chat template is not applied` — BF16도 같은 경고를 받았고 정상 진행됨, 네 기법 모두 동일 조건이라 비교에는 영향 없음
