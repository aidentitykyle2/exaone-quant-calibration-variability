#!/usr/bin/env python3
"""
find_revision.py — 현재 transformers와 호환되는 모델 리비전 찾기

EXAONE 저장소의 원격 코드(configuration_exaone.py, modeling_exaone.py)가 최신
transformers용으로 갱신되면, 고정된 구버전 transformers에서는 import 오류가 난다.
실제로 3.5-32B의 main 리비전이 RopeParameters를 import해 4.57.6에서 실패했다.

이 스크립트는 저장소의 커밋을 최신순으로 훑으며 각 리비전을 별도 프로세스에서
시험한다. 같은 프로세스에서 여러 리비전의 원격 코드를 import하면 모듈 이름이
충돌하므로 반드시 분리 실행한다.

검사 내용
  1) AutoConfig 로드
  2) 빈 가중치로 모델 구성 → 모듈 이름 규약 확인 (AWQ·QuaRot 매핑 재사용 가능 여부)

사용법:
  python find_revision.py --model LGAI-EXAONE/EXAONE-3.5-32B-Instruct
  python find_revision.py --model ... --max-commits 30
"""

import argparse
import json
import subprocess
import sys

PROBE = r'''
import sys, re, json
from transformers import AutoConfig, AutoModelForCausalLM
from accelerate import init_empty_weights
mid, rev = sys.argv[1], sys.argv[2]
NEEDED = ["ln_1", "attn.attention.q_proj", "attn.attention.out_proj",
          "mlp.c_fc_0", "mlp.c_fc_1", "mlp.c_proj", "ln_2"]
try:
    c = AutoConfig.from_pretrained(mid, revision=rev, trust_remote_code=True)
    with init_empty_weights():
        # from_config는 revision을 클래스 탐색과 생성자 양쪽에 넘겨 생성자에서 오류가 난다.
        # 클래스 탐색만 리비전을 지정해 따로 수행한 뒤 생성한다.
        ref = (getattr(c, "auto_map", None) or {}).get("AutoModelForCausalLM")
        if ref:
            from transformers.dynamic_module_utils import get_class_from_dynamic_module
            m = get_class_from_dynamic_module(ref, mid, revision=rev)(c)
        else:
            m = AutoModelForCausalLM.from_config(c, trust_remote_code=True)
    names = [n for n, _ in m.named_modules()]
    layer0 = [n for n in names if re.search(r"\.h\.0\.", n)]
    missing = [k for k in NEEDED if not any(n.endswith(k) for n in layer0)]
    print("RESULT " + json.dumps({
        "ok": True, "missing": missing,
        "layers": c.num_hidden_layers, "hidden": c.hidden_size,
        "heads": c.num_attention_heads, "kv": c.num_key_value_heads,
        "intermediate": getattr(c, "intermediate_size", None)}))
except Exception as e:
    print("RESULT " + json.dumps({"ok": False, "err": f"{type(e).__name__}: {e}"[:200]}))
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--max-commits", type=int, default=20)
    args = ap.parse_args()

    from huggingface_hub import list_repo_commits
    commits = list_repo_commits(args.model)[: args.max_commits]
    print(f"{args.model}: 커밋 {len(commits)}개 확인 (최신순)\n")

    good = None
    for i, c in enumerate(commits):
        sha = c.commit_id
        date = str(getattr(c, "created_at", ""))[:10]
        title = (c.title or "")[:45]
        p = subprocess.run([sys.executable, "-c", PROBE, args.model, sha],
                           capture_output=True, text=True, timeout=600)
        line = next((l for l in p.stdout.splitlines() if l.startswith("RESULT ")), None)
        r = json.loads(line[7:]) if line else {"ok": False, "err": "출력 없음"}
        if r["ok"]:
            tag = "모듈 규약 일치" if not r["missing"] else f"모듈 불일치 {r['missing']}"
            print(f"  [{i:2d}] {sha[:12]} {date} | 통과 | {tag}")
            print(f"       layers {r['layers']} hidden {r['hidden']} "
                  f"heads {r['heads']}/{r['kv']} inter {r['intermediate']}")
            if good is None and not r["missing"]:
                good = sha
                break
        else:
            print(f"  [{i:2d}] {sha[:12]} {date} | 실패 | {r['err'][:90]}")
        if title:
            print(f"       ({title})")

    print()
    if good:
        print(f"사용할 리비전: {good}")
        print("\n적용 방법:")
        print(f'  REV32="{good}" 를 run_72h.sh 에 설정하거나 아래처럼 직접 지정')
        print(f'  python run_quant_stage.py --stage gptq --model {args.model} \\')
        print(f'    --revision {good} --calib-lang ko --calib-n 8 --calib-seed 1')
    else:
        print("호환되는 리비전을 찾지 못했다.")
        print("대안 1) --max-commits 를 늘려 더 과거까지 훑는다")
        print("대안 2) 4.0-32B를 쓴다 (공식 통합 모델이라 원격 코드가 없어 이 문제가 없다)")


if __name__ == "__main__":
    main()
