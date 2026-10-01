#!/bin/bash
# preflight_32b.sh — EXAONE 3.5-32B 양자화 가능 여부 사전 점검
#
# 52시간 배치를 걸기 전에, 실패 가능한 지점을 싼 것부터 확인한다.
#
#   quick (기본, 약 2분, 가중치 다운로드 없음)
#     1 설정        레이어·크기 확인
#     2 모듈 구조   빈 가중치로 모델을 구성해 모듈 이름이 7.8B 규약과 같은지 확인
#                   (AWQ·QuaRot 매핑이 그대로 통하는지가 여기서 드러난다)
#     3 토크나이저
#
#   full (약 40~90분, 가중치 약 60 GiB 다운로드 포함)
#     4 실제 GPTQ 양자화 (캘리브레이션 8개) + vLLM 로드
#       GPTQ는 아키텍처별 매핑을 쓰지 않으므로, 여기까지 통과하면 본 실험이 가능하다.
#
# 사용법:
#   bash preflight_32b.sh            # quick
#   bash preflight_32b.sh full       # 실제 양자화까지 (GPU 1장)

set -u
MID="${MID:-LGAI-EXAONE/EXAONE-3.5-32B-Instruct}"
# 리비전을 비워 두면 main을 쓴다. main의 원격 코드가 갱신되어 구버전 transformers에서
# 깨질 수 있으므로, find_revision.py 로 호환 리비전을 찾아 지정하는 것이 안전하다.
REV="${REV:-d6fa88cd8d2c9512b40578bdc44e64909e5a5042}"
REF_REV="${REF_REV:-0ff6b5ec7c13b049b253a16a889aa269e6b79a94}"
REF="${REF:-LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct}"
MODE="${1:-quick}"
GPU="${CUDA_VISIBLE_DEVICES:-0}"
[ -z "${VIRTUAL_ENV:-}" ] && { echo "venv 미활성화"; exit 1; }
echo "점검 대상: ${MID}   (기준 모델 ${REF})"
echo ""

echo "=== 1·2. 설정과 모듈 구조 (가중치 다운로드 없음) ==="
CUDA_VISIBLE_DEVICES="" python - "$MID" "$REV" "$REF" "$REF_REV" <<'PY'
import re, sys
from transformers import AutoConfig, AutoModelForCausalLM
from accelerate import init_empty_weights

# 7.8B에서 검증된 매핑이 기대하는 모듈 이름 규약
NEEDED = ["ln_1", "attn.attention.q_proj", "attn.attention.out_proj",
          "mlp.c_fc_0", "mlp.c_fc_1", "mlp.c_proj", "ln_2"]

def probe(mid, rev):
    c = AutoConfig.from_pretrained(mid, revision=rev or None, trust_remote_code=True)
    with init_empty_weights():
        # from_config는 revision을 클래스 탐색과 생성자 양쪽에 넘겨 생성자에서 오류가 난다.
        # 클래스 탐색만 리비전을 지정해 따로 수행한 뒤 생성한다.
        ref = (getattr(c, "auto_map", None) or {}).get("AutoModelForCausalLM")
        if ref:
            from transformers.dynamic_module_utils import get_class_from_dynamic_module
            m = get_class_from_dynamic_module(ref, mid, revision=rev or None)(c)
        else:
            m = AutoModelForCausalLM.from_config(c, trust_remote_code=True)
    names = [n for n, _ in m.named_modules()]
    return c, names

bad = 0
ref_names = None
pairs = list(zip(sys.argv[1::2], sys.argv[2::2]))
for mid, rev in pairs:
    try:
        c, names = probe(mid, rev)
        inter = getattr(c, "intermediate_size", None)
        p2 = inter is not None and (inter & (inter - 1)) == 0
        print(f"  {mid}")
        print(f"    layers {c.num_hidden_layers} | hidden {c.hidden_size} | "
              f"heads {c.num_attention_heads}/{c.num_key_value_heads} | inter {inter} (2^n: {p2})")
        layer0 = [n for n in names if re.search(r"\.h\.0\.", n)]
        missing = [k for k in NEEDED if not any(n.endswith(k) for n in layer0)]
        if missing:
            print(f"    모듈 규약 불일치 — 없는 이름: {missing}")
            print(f"    실제 예시: {[n.split('.h.0.')[-1] for n in layer0][:8]}")
            bad = 1
        else:
            print(f"    모듈 규약 일치 (AWQ·QuaRot 매핑 재사용 가능)")
    except Exception as e:
        print(f"  {mid} 실패: {type(e).__name__}: {e}")
        bad = 1
sys.exit(bad)
PY
RC1=$?
[ ${RC1} -eq 0 ] && echo "  [통과] 설정·모듈 구조" || echo "  [실패] 설정 또는 모듈 구조"
echo ""

echo "=== 3. 토크나이저 ==="
CUDA_VISIBLE_DEVICES="" python - "$MID" <<'PY'
import sys
from transformers import AutoTokenizer
t = AutoTokenizer.from_pretrained(sys.argv[1], trust_remote_code=True)
print(f"  vocab {len(t)} | 한국어 토큰화 {t.tokenize('한국어 양자화 실험')[:4]}")
PY
RC2=$?
[ ${RC2} -eq 0 ] && echo "  [통과] 토크나이저" || echo "  [실패] 토크나이저"
echo ""

echo "=== 캐시·디스크 ==="
CACHE="${HF_HOME:-$HOME/.cache/huggingface}/hub/models--${MID//\//--}"
if [ -d "${CACHE}" ]; then
  echo "  캐시 있음: $(du -sh "${CACHE}" 2>/dev/null | cut -f1)"
else
  echo "  캐시 없음 — full 모드는 약 60 GiB를 내려받는다"
fi
df -h /data 2>/dev/null | tail -1 | sed 's/^/  여유: /'
echo ""

if [ "${MODE}" != "full" ]; then
  echo "=== 요약 (quick) ==="
  if [ ${RC1} -eq 0 ] && [ ${RC2} -eq 0 ]; then
    echo "  구조 점검 통과. 실제 양자화까지 확인하려면: bash preflight_32b.sh full"
    echo "  GPTQ는 매핑을 쓰지 않으므로 구조가 같으면 대체로 동작한다."
    exit 0
  fi
  echo "  구조 점검 실패. full을 돌리지 말 것."
  exit 1
fi

echo "=== 4. 실제 GPTQ 양자화 (캘리브레이션 8개) + vLLM 로드 ==="
echo "    가중치 다운로드가 필요하면 시간이 더 걸린다. 산출물은 버려도 된다."
date
CUDA_VISIBLE_DEVICES="${GPU}" python run_quant_stage.py --stage gptq --model "${MID}" \
  ${REV:+--revision ${REV}} --calib-lang ko --calib-n 8 --calib-seed 1 2>&1 | tail -30
RC3=${PIPESTATUS[0]}
date
echo ""
echo "=== 요약 (full) ==="
if [ ${RC1} -eq 0 ] && [ ${RC2} -eq 0 ] && [ ${RC3} -eq 0 ]; then
  echo "  전부 통과. 72시간 배치를 실행해도 된다: nohup bash run_72h.sh > run72.txt 2>&1 & disown"
  echo "  위 양자화 소요 시간에 32배를 곱하면 n=256 한 건의 대략적 전파 시간이 된다."
  exit 0
fi
echo "  실패 (설정 ${RC1} / 토크나이저 ${RC2} / 양자화 ${RC3})"
echo "  3.5-32B가 안 되면 대안: 4.0-32B는 1.2B 사전 점검에서 파이프라인이 검증되었으나,"
echo "  세대가 달라 7.8B와의 비교에 아키텍처 교란이 생긴다는 점을 논문에 명시해야 한다."
exit 1
