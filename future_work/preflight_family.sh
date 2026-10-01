#!/bin/bash
# preflight_family.sh — 다른 모델 계열로 확장하기 전 사전 점검 (모델 무관)
#
# *** 향후 연구용이다. 본 논문의 결과에는 사용되지 않았다. ***
#
# 본 논문의 한계는 "모델 계열이 하나(EXAONE)"라는 점이다. 이를 메우려면
# 다른 계열에서 같은 설계를 반복해야 하는데, 36회 양자화(약 20 GPU시간)를
# 걸기 전에 실패 지점과 **측정 가능성**을 먼저 확인해야 한다.
#
#   quick (기본, 약 3분, 가중치 다운로드 없음)
#     1 저장소 존재·리비전   구버전 transformers(4.57.6)와 호환되는 리비전 확인
#     2 설정·모듈 구조       GPTQ는 매핑을 쓰지 않으므로 Linear 층 존재만 확인.
#                            AWQ/QuaRot까지 확장할 계획이면 매핑 필요 여부도 표시.
#     3 토크나이저           한국어 토큰화가 바이트 단위로 부서지지 않는지 확인
#   gate (약 1~2시간, 가중치 다운로드 + BF16 평가)
#     4 BF16 KMMLU/MMLU      ** 가장 중요한 관문 **
#        KMMLU 정확도가 우연 수준(4지선다 0.25)에 가까우면, 한국어 성능
#        "저하율"을 정의할 분모가 없어 언어 비대칭 지표 X 자체가 무의미해진다.
#        본 논문의 EXAONE 3.5 기준: 7.8B KMMLU 0.4555, 32B 0.4990.
#        최소 기준을 --min-kmmlu(기본 0.33)로 둔다.
#   full (추가 약 30~60분)
#     5 실제 GPTQ 양자화(캘리브레이션 8개) + vLLM 로드
#
# 사용법:
#   bash preflight_family.sh                     # quick
#   MID=Qwen/Qwen2.5-7B-Instruct bash preflight_family.sh gate
#   bash preflight_family.sh full
#
# 후보 모델에 대한 메모 (저장소 이름은 반드시 1단계로 실재를 확인할 것):
#   - Qwen/Qwen2.5-7B-Instruct
#       다국어 모델. 한국어 중심은 아니다. EXAONE과 대비되는 축이 되어,
#       "한국어 중심 사전학습이 비대칭의 조건인가"를 물을 수 있다. 공식
#       transformers 지원이라 원격 코드·리비전 고정 문제가 없다. 기본값.
#   - 한국어 중심 계열(Kanana, Mi:dm, A.X 등)
#       EXAONE과 같은 축에서의 반복 검증이 된다. 다만 저장소 이름·공개 여부·
#       라이선스가 본 작성 시점 이후 바뀌었을 수 있으므로 1단계로 확인할 것.
#   - meta-llama/Llama-3.1-8B-Instruct
#       영어 중심 대조군. 한국어 성능이 4단계 관문을 넘는지 특히 주의해서 볼 것.
#
# 어느 쪽을 고르든, 결론의 성격이 달라진다는 점을 논문에 명시해야 한다:
#   한국어 중심 계열 추가 -> 본 발견의 반복 검증
#   비한국어 중심 계열 추가 -> 사전학습 언어 구성이 조건인지에 대한 대비

set -u
MID="${MID:-Qwen/Qwen2.5-7B-Instruct}"
REV="${REV:-}"                 # 비우면 main. 원격 코드 모델은 find_revision.py로 찾을 것
MODE="${1:-quick}"
GPU="${CUDA_VISIBLE_DEVICES:-0}"
GPU_UTIL="${GPU_UTIL:-0.70}"
MIN_KMMLU="${MIN_KMMLU:-0.33}"
OUTDIR="${OUTDIR:-preflight_family/$(basename "${MID}")}"
[ -z "${VIRTUAL_ENV:-}" ] && { echo "venv 미활성화"; exit 1; }
mkdir -p "${OUTDIR}"

echo "점검 대상: ${MID}  (리비전 ${REV:-main})"
echo "산출물   : ${OUTDIR}"
echo ""

echo "=== 1. 저장소 존재와 리비전 호환성 ==="
CUDA_VISIBLE_DEVICES="" python - "$MID" "$REV" <<'PY'
import sys
mid, rev = sys.argv[1], (sys.argv[2] or None)
from huggingface_hub import HfApi
api = HfApi()
try:
    info = api.model_info(mid, revision=rev)
except Exception as e:
    print(f"  저장소를 찾을 수 없다: {type(e).__name__}: {e}")
    print("  이름이 맞는지, 비공개·게이트 모델이라 토큰이 필요한지 확인할 것.")
    sys.exit(1)
files = [s.rfilename for s in (info.siblings or [])]
remote = [f for f in files if f.endswith(".py")]
print(f"  존재 확인 | sha {info.sha[:12]} | 파일 {len(files)}개")
print(f"  원격 코드 파일: {remote if remote else '없음 (transformers 공식 지원 — 리비전 고정 불필요)'}")
if remote:
    print("  주의: 원격 코드 모델은 저장소가 갱신되면 구버전 transformers에서 깨질 수 있다.")
    print("        find_revision.py 로 호환 리비전을 찾아 REV로 고정할 것.")
PY
RC1=$?
[ ${RC1} -eq 0 ] && echo "  [통과] 저장소" || echo "  [실패] 저장소"
echo ""

echo "=== 2. 설정과 모듈 구조 (가중치 다운로드 없음) ==="
CUDA_VISIBLE_DEVICES="" python - "$MID" "$REV" <<'PY'
import re, sys
from transformers import AutoConfig, AutoModelForCausalLM
from accelerate import init_empty_weights
mid, rev = sys.argv[1], (sys.argv[2] or None)
c = AutoConfig.from_pretrained(mid, revision=rev, trust_remote_code=True)
with init_empty_weights():
    ref = (getattr(c, "auto_map", None) or {}).get("AutoModelForCausalLM")
    if ref:
        from transformers.dynamic_module_utils import get_class_from_dynamic_module
        m = get_class_from_dynamic_module(ref, mid, revision=rev)(c)
    else:
        m = AutoModelForCausalLM.from_config(c, trust_remote_code=True)
import torch
names = [n for n, mod in m.named_modules() if isinstance(mod, torch.nn.Linear)]
inter = getattr(c, "intermediate_size", None)
print(f"  layers {c.num_hidden_layers} | hidden {c.hidden_size} | "
      f"heads {c.num_attention_heads}/{getattr(c, 'num_key_value_heads', '?')} | inter {inter}")
print(f"  Linear 층 {len(names)}개 -> GPTQ는 아키텍처별 매핑 없이 동작한다")
# Hessian 메모리 추정: 가장 큰 입력 차원이 상한을 정한다
big = max([c.hidden_size] + ([inter] if inter else []))
print(f"  Hessian 상한 ≈ {big}^2 x 4B = {big*big*4/2**30:.2f} GiB (강건 집계는 K배)")
# AWQ/QuaRot까지 확장할 경우에만 필요한 정보
layer0 = [n for n in names if re.search(r"\.(0)\.", n)]
print(f"  층 0 Linear 이름 예: {[n.split('.')[-2] + '.' + n.split('.')[-1] for n in layer0][:8]}")
llama_like = all(any(n.endswith(k) for n in layer0)
                 for k in ["q_proj", "k_proj", "v_proj", "o_proj",
                           "gate_proj", "up_proj", "down_proj"])
print(f"  Llama 명명 규약 일치: {llama_like} "
      f"({'AWQ/QuaRot 자동 추론 사용 가능' if llama_like else 'AWQ/QuaRot 확장 시 수동 매핑 필요'})")
PY
RC2=$?
[ ${RC2} -eq 0 ] && echo "  [통과] 설정·모듈 구조" || echo "  [실패] 설정·모듈 구조"
echo ""

echo "=== 3. 토크나이저 (한국어 처리) ==="
CUDA_VISIBLE_DEVICES="" python - "$MID" "$REV" <<'PY'
import sys
from transformers import AutoTokenizer
mid, rev = sys.argv[1], (sys.argv[2] or None)
t = AutoTokenizer.from_pretrained(mid, revision=rev, trust_remote_code=True)
ko = "한국어 양자화 실험에서 캘리브레이션 표본의 언어가 성능에 영향을 주는가"
en = "Does the language of calibration samples affect quantized model accuracy"
tk, te = t.tokenize(ko), t.tokenize(en)
print(f"  vocab {len(t)}")
print(f"  한국어 {len(ko)}자 -> {len(tk)}토큰 (문자/토큰 {len(ko)/max(len(tk),1):.2f})")
print(f"  영어   {len(en)}자 -> {len(te)}토큰 (문자/토큰 {len(en)/max(len(te),1):.2f})")
print(f"  한국어 토큰 예: {tk[:6]}")
# 문자/토큰이 1.0에 가까우면 한국어를 사실상 바이트 단위로 쪼개고 있다는 뜻이다.
# 그 경우 같은 표본 수(256개 문장)가 담는 한국어 정보량이 EXAONE과 크게 달라져,
# "표본 수 고정"이라는 통제가 느슨해진다 — 논문에 반드시 밝힐 것.
r = len(ko) / max(len(tk), 1)
if r < 1.2:
    print("  경고: 한국어 문자/토큰 비가 낮다(바이트 수준 분절 가능성).")
    print("        같은 문장 수라도 한국어 토큰 예산이 EXAONE과 달라지므로,")
    print("        표본 수 대신 토큰 수를 맞추는 설계를 고려하고 논문에 명시할 것.")
PY
RC3=$?
[ ${RC3} -eq 0 ] && echo "  [통과] 토크나이저" || echo "  [실패] 토크나이저"
echo ""

if [ "${MODE}" = "quick" ]; then
  echo "=== 요약 (quick) ==="
  if [ ${RC1} -eq 0 ] && [ ${RC2} -eq 0 ] && [ ${RC3} -eq 0 ]; then
    echo "  구조 점검 통과. 다음은 측정 가능성 관문이다:"
    echo "    bash preflight_family.sh gate"
    exit 0
  fi
  echo "  구조 점검 실패. 다음 단계로 넘어가지 말 것."
  exit 1
fi

echo "=== 4. BF16 KMMLU/MMLU 관문 (측정 가능성) ==="
echo "    KMMLU가 우연 수준(0.25)에 가까우면 언어 비대칭 지표를 정의할 수 없다."
date
REV_ARG=""
[ -n "${REV}" ] && REV_ARG=",revision=${REV}"
COMMON="trust_remote_code=True,gpu_memory_utilization=${GPU_UTIL},max_model_len=2048"
HF_HUB_OFFLINE=0 CUDA_VISIBLE_DEVICES="${GPU}" lm_eval --model vllm \
  --model_args "pretrained=${MID},dtype=bfloat16${REV_ARG},${COMMON}" \
  --tasks kmmlu,mmlu --num_fewshot 5 --batch_size auto \
  --output_path "${OUTDIR}/bf16_gate" 2>&1 | tail -25
RC4=${PIPESTATUS[0]}
date
CUDA_VISIBLE_DEVICES="" python - "${OUTDIR}/bf16_gate" "${MIN_KMMLU}" <<'PY'
import json, sys, pathlib
root, thr = pathlib.Path(sys.argv[1]), float(sys.argv[2])
files = sorted(root.rglob("results*.json"))
if not files:
    print("  결과 파일을 찾지 못했다."); sys.exit(1)
res = json.loads(files[-1].read_text())["results"]
def acc(task):
    r = res.get(task, {})
    for k in ("acc_norm,none", "acc,none", "acc_norm", "acc"):
        if k in r: return float(r[k])
    return None
ko, en = acc("kmmlu"), acc("mmlu")
print(f"  KMMLU {ko}  MMLU {en}  (EXAONE 3.5-7.8B 기준 0.4555 / 참고)")
if ko is None:
    print("  KMMLU 점수를 읽지 못했다."); sys.exit(1)
if ko < thr:
    print(f"  [실패] KMMLU {ko:.4f} < 기준 {thr}. 한국어 성능이 낮아 저하율의 분모가")
    print("         불안정하다. 이 모델로는 언어 비대칭을 신뢰성 있게 측정할 수 없다.")
    print("         다른 계열을 고르거나, 한국어 벤치마크를 KoBEST 쪽으로 바꿀 것.")
    sys.exit(1)
print(f"  [통과] KMMLU {ko:.4f} >= 기준 {thr}")
if en and ko: print(f"  참고: 언어 간 절대 격차 {abs(en-ko):.4f} — 클수록 두 언어의")
if en and ko: print(f"        문항 난이도 차이가 X 해석에 섞이므로 논문에 함께 보고할 것.")
PY
RC5=$?
echo ""

if [ "${MODE}" != "full" ]; then
  echo "=== 요약 (gate) ==="
  if [ ${RC4} -eq 0 ] && [ ${RC5} -eq 0 ]; then
    echo "  측정 가능성 관문 통과. 실제 양자화까지 확인하려면: bash preflight_family.sh full"
    exit 0
  fi
  echo "  관문 실패 (평가 ${RC4} / 기준 ${RC5}). 본 실험을 걸지 말 것."
  exit 1
fi

echo "=== 5. 실제 GPTQ 양자화 (캘리브레이션 8개) + vLLM 로드 ==="
date
CUDA_VISIBLE_DEVICES="${GPU}" python run_quant_stage.py --stage gptq --model "${MID}" \
  ${REV:+--revision ${REV}} --calib-lang ko --calib-n 8 --calib-seed 1 2>&1 | tail -30
RC6=${PIPESTATUS[0]}
date
echo ""
echo "=== 요약 (full) ==="
if [ ${RC1} -eq 0 ] && [ ${RC2} -eq 0 ] && [ ${RC3} -eq 0 ] \
   && [ ${RC4} -eq 0 ] && [ ${RC5} -eq 0 ] && [ ${RC6} -eq 0 ]; then
  echo "  전부 통과. 본 실험을 실행해도 된다:"
  echo "    MID=${MID} nohup bash run_family.sh > run_family.txt 2>&1 & disown"
  echo "  위 8개 양자화 소요 시간에 32배를 곱하면 n=256 한 건의 대략적 소요가 된다."
  exit 0
fi
echo "  실패 (저장소 ${RC1} / 구조 ${RC2} / 토크나이저 ${RC3} / 평가 ${RC4} / 기준 ${RC5} / 양자화 ${RC6})"
exit 1
