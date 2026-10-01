#!/usr/bin/env python3
"""
apply_fixes.py — 7.8B 하드코딩을 푸는 수정을 멱등하게 적용한다

patch(1)가 "Reversed (or previously applied)"로 막히는 상황을 피하려고 만들었다.
이미 적용된 파일은 건드리지 않고 건너뛰며, 적용 전에 .bak 사본을 남기고
적용 후 문법 검사를 한다. 몇 번 돌려도 결과가 같다.

고치는 것
  activation_analysis.py  MODEL_ID / REVISION 을 환경변수로 덮어쓸 수 있게 한다.
                          ppl_eval.py가 이 두 값을 import해 BF16 기준 모델을
                          정하므로, 하드코딩 상태로 32B를 재면 양자화는 32B인데
                          BF16만 7.8B가 되어 PPL 증가율이 무의미해진다.
  rescore_all.py          정합성 점검 참조값을 하드코딩(7.8B) 대신 그 실행의
                          results*.json에서 읽는다.
  make_figure1.py         하드코딩된 7.8B X 값 대신 CLI 인자를 받는다.

사용법
  python apply_fixes.py            # 무엇을 바꿀지 보여주기만 한다
  python apply_fixes.py --write    # 실제로 적용한다
  python apply_fixes.py --revert   # .bak 으로 되돌린다
"""

from __future__ import annotations

import argparse
import ast
import shutil
import sys
from pathlib import Path

# ---------------------------------------------------------------- 1
AA_OLD = '''MODEL_ID = "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct"
REVISION = "0ff6b5ec7c13b049b253a16a889aa269e6b79a94"'''

AA_NEW = '''# 모델을 환경변수로 덮어쓸 수 있게 한다. ppl_eval.py가 이 두 값을 import해
# BF16 기준 모델과 토크나이저를 정하므로, 하드코딩된 상태에서 32B를 측정하면
# 양자화 체크포인트는 32B인데 BF16 기준만 7.8B가 되어 PPL 증가율이 무의미해진다.
#   QUANT_MODEL_ID=LGAI-EXAONE/EXAONE-3.5-32B-Instruct \\
#   QUANT_REVISION=d6fa88cd8d2c9512b40578bdc44e64909e5a5042 python ppl_eval.py measure ...
# QUANT_REVISION을 빈 문자열로 두면 리비전을 고정하지 않는다.
MODEL_ID = os.environ.get("QUANT_MODEL_ID",
                          "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct")
REVISION = os.environ.get("QUANT_REVISION",
                          "0ff6b5ec7c13b049b253a16a889aa269e6b79a94") or None'''

# ---------------------------------------------------------------- 2
RA_OLD_1 = '''# 기존 lm-eval 보고치 (정합성 점검용, first 기준)
REPORTED = {("bf16", "kmmlu"): 0.4555, ("bf16", "haerae"): 0.7663, ("bf16", "mmlu"): 0.6585,
            ("bf16", "kobest_boolq"): 0.9053, ("bf16", "boolq"): 0.8648}'''

RA_NEW_1 = '''# 정합성 점검용 참조값.
# 이전 버전은 7.8B의 보고치를 하드코딩해 두어, 다른 모델(32B 등)을 점검하면
# 항상 "불일치"가 떴다. 점검의 목적은 "그 실행에서 lm-eval이 스스로 적은 값"과
# 우리가 재계산한 first 점수가 같은지 보는 것이므로, 각 평가 디렉터리의
# results*.json에서 직접 읽는다. 이렇게 하면 모델이 바뀌어도 그대로 쓸 수 있다.
REPORTED_FALLBACK = {   # results json이 없을 때만 쓰고, 7.8B 전용임을 명시한다.
    "EXAONE-3.5-7.8B-Instruct": {
        "kmmlu": 0.4555, "haerae": 0.7663, "mmlu": 0.6585,
        "kobest_boolq": 0.9053, "boolq": 0.8648,
    },
}


def reported_from_results(model_dir: Path, task_names) -> dict:
    """lm-eval이 그 실행에서 기록한 집계 정확도를 과제별로 읽는다."""
    files = sorted(model_dir.rglob("results*.json"), key=lambda p: p.stat().st_mtime)
    if not files:
        return {}
    try:
        res = json.loads(files[-1].read_text(encoding="utf-8")).get("results", {})
    except Exception:
        return {}
    out = {}
    for n in task_names:
        r = res.get(n) or {}
        for k in ("acc_norm,none", "acc,none", "acc_norm", "acc"):
            if k in r:
                try:
                    out[n] = float(r[k])
                except (TypeError, ValueError):
                    pass
                break
    return out'''

RA_OLD_2 = '''    data = {}   # (lang, model) -> 결과
    lines = []'''
RA_NEW_2 = '''    data = {}       # (lang, model) -> 결과
    reported = {}   # (lang, model) -> lm-eval이 그 실행에서 기록한 정확도
    lines = []'''

RA_OLD_3 = '''            res, mism = score_dir(d, lang)
            if res:
                data[(lang, d.name)] = res'''
RA_NEW_3 = '''            res, mism = score_dir(d, lang)
            if res:
                data[(lang, d.name)] = res
                reported[(lang, d.name)] = reported_from_results(
                    d, [n for n, _, _ in TASKS[lang]])'''

RA_OLD_4 = '''    lines.append("=== 정합성 점검: first 기준이 기존 lm-eval 보고치와 일치하는가 ===")
    for (m, t), rep in REPORTED.items():
        lang = "ko" if t in dict((n, 1) for n, _, _ in TASKS["ko"]) else "en"
        got = data.get((lang, m), {}).get(t, {}).get("first")
        if got is not None:
            lines.append(f"  {m:6s} {t:14s} 보고치 {rep:.4f}  재계산 {got:.4f}  "
                         f"{'일치' if abs(got - rep) < 0.0006 else '불일치 — 확인 필요'}")'''

RA_NEW_4 = '''    lines.append("=== 정합성 점검: first 기준이 그 실행의 lm-eval 보고치와 일치하는가 ===")
    lines.append(f"    참조: {base.name} 의 각 seed_eval/*/<조건>/results*.json")
    n_checked = 0
    for (lang, m), rep_map in sorted(reported.items()):
        for t, rep in sorted(rep_map.items()):
            got = data.get((lang, m), {}).get(t, {}).get("first")
            if got is None:
                continue
            n_checked += 1
            ok = abs(got - rep) < 0.0006
            if ok and m != "bf16":
                continue        # 정상인 양자화 조건은 줄 수가 많아 생략한다
            lines.append(f"  {m:24s} {t:14s} 보고치 {rep:.4f}  재계산 {got:.4f}  "
                         f"{'일치' if ok else '불일치 — 확인 필요'}")
    if n_checked == 0:
        fb = REPORTED_FALLBACK.get(base.name)
        if fb:
            lines.append("    results json을 찾지 못해 내장 참조값으로 대조한다(7.8B 전용).")
            for t, rep in fb.items():
                lang = "ko" if t in dict((n, 1) for n, _, _ in TASKS["ko"]) else "en"
                got = data.get((lang, "bf16"), {}).get(t, {}).get("first")
                if got is not None:
                    lines.append(f"  {'bf16':24s} {t:14s} 보고치 {rep:.4f}  재계산 {got:.4f}  "
                                 f"{'일치' if abs(got - rep) < 0.0006 else '불일치 — 확인 필요'}")
        else:
            lines.append(f"    대조할 참조값이 없다({base.name}). 점검을 건너뛴다.")
    else:
        lines.append(f"    대조한 (조건, 과제) 쌍 {n_checked}개. 불일치만 위에 표시(bf16은 항상 표시).")'''

# ---------------------------------------------------------------- 3
MF_OLD_1 = '''import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

X = [3.29,-2.45,0.71,2.15,3.03,-0.58,0.41,-0.44,1.20,2.89,
     2.46,0.01,0.06,-1.24,1.14,-0.41,5.74,2.94]
SINGLE = (3.27, 2.20, 4.33)     # seed 0, 문항만
MULTI  = (1.20, 0.12, 2.16)     # 18표본, 문항+표본'''

MF_NEW_1 = '''import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

# 7.8B 기본값. 다른 모델(32B 등)은 --x / --single / --multi 또는 --json으로 넘긴다.
#   single_run_risk.py -> SINGLE (단일 실행, 문항만 재표집)
#   final_analysis.py  -> MULTI  (두 단계: 문항+표본)
X_DEFAULT = [3.29,-2.45,0.71,2.15,3.03,-0.58,0.41,-0.44,1.20,2.89,
             2.46,0.01,0.06,-1.24,1.14,-0.41,5.74,2.94]
SINGLE_DEFAULT = (3.27, 2.20, 4.33)
MULTI_DEFAULT  = (1.20, 0.12, 2.16)

_ap = argparse.ArgumentParser(description="논문 그림: 캘리브레이션 표본별 언어 비대칭")
_ap.add_argument("--x", nargs="+", type=float, help="시드별 X 값")
_ap.add_argument("--single", nargs=3, type=float, metavar=("M", "LO", "HI"),
                 help="단일 실행 구간 (평균 하한 상한)")
_ap.add_argument("--multi", nargs=3, type=float, metavar=("M", "LO", "HI"),
                 help="표본 반영 구간 (평균 하한 상한)")
_ap.add_argument("--json", help='{"x": [...], "single": [m,lo,hi], "multi": [m,lo,hi]} 파일')
_ap.add_argument("--out", default="figure1_asymmetry", help="출력 파일 이름(확장자 제외)")
_ap.add_argument("--label", default=None, help="제목에 덧붙일 모델 이름(예: EXAONE 3.5-32B)")
_a = _ap.parse_args()

if _a.json:
    _d = json.loads(open(_a.json, encoding="utf-8").read())
    X = list(map(float, _d["x"]))
    SINGLE = tuple(map(float, _d["single"]))
    MULTI = tuple(map(float, _d["multi"]))
else:
    X = _a.x if _a.x else X_DEFAULT
    SINGLE = tuple(_a.single) if _a.single else SINGLE_DEFAULT
    MULTI = tuple(_a.multi) if _a.multi else MULTI_DEFAULT
if _a.x and not (_a.single and _a.multi):
    raise SystemExit("--x를 주면 --single과 --multi도 함께 주어야 한다. "
                     "구간은 single_run_risk.py와 final_analysis.py 출력에서 가져올 것.")'''

MF_OLD_2 = '''             single="단일 실행\\n(문항만 재표집)", multi="18표본\\n(문항+표본)",'''
MF_NEW_2 = '''             single="단일 실행\\n(문항만 재표집)", multi=f"{len(X)}표본\\n(문항+표본)",'''

MF_OLD_3 = '''             single="Single run\\n(items only)", multi="18 samples\\n(items+samples)",'''
MF_NEW_3 = '''             single="Single run\\n(items only)", multi=f"{len(X)} samples\\n(items+samples)",'''

MF_OLD_4 = '''ax.set_xlabel(L["xl"]); ax.set_ylabel(L["yl"]); ax.set_title(L["ttl"], fontsize=11)'''
MF_NEW_4 = '''_ttl = L["ttl"] + (f"  [{_a.label}]" if _a.label else "")
ax.set_xlabel(L["xl"]); ax.set_ylabel(L["yl"]); ax.set_title(_ttl, fontsize=11)'''

MF_OLD_5 = '''for ext in ("png", "pdf"):
    fig.savefig(f"figure1_asymmetry.{ext}", dpi=200, bbox_inches="tight")
print(f"저장: figure1_asymmetry.png / .pdf  (한글 폰트 {'사용' if ko else '없음 → 영문 라벨'})")'''
MF_NEW_5 = '''for ext in ("png", "pdf"):
    fig.savefig(f"{_a.out}.{ext}", dpi=200, bbox_inches="tight")
print(f"저장: {_a.out}.png / .pdf  (한글 폰트 {'사용' if ko else '없음 → 영문 라벨'})")'''


# ---------------------------------------------------------------- 4
# 32B에서 드러난 OOM. 4비트 체크포인트를 transformers로 열면 조밀 텐서로 풀리는데
# (dequant_value * scale), 32B는 그 결과가 60GiB를 넘어 한 장에 들어가지 않는다.
#   - 양자화 경로의 dtype을 "auto"에서 bfloat16으로 못 박아 fp32로 풀릴 여지를 없앤다
#   - QUANT_DEVICE_MAP=auto 로 여러 카드에 나눠 올릴 수 있게 한다
#   - QUANT_MAX_MEMORY="0=40GiB,1=40GiB,cpu=80GiB" 로 카드별 상한을 줄 수 있게 한다
AA2_OLD = '''def load_model(path_or_id, is_bf16):
    from transformers import AutoModelForCausalLM
    kw = dict(trust_remote_code=True, device_map="cuda:0")
    if is_bf16:
        kw.update(revision=REVISION, torch_dtype=torch.bfloat16)
    else:
        kw.update(torch_dtype="auto")
    m = AutoModelForCausalLM.from_pretrained(path_or_id, **kw)
    m.eval()
    return m'''

AA2_NEW = '''def load_model(path_or_id, is_bf16):
    """
    QUANT_DEVICE_MAP 으로 배치를 바꿀 수 있다(기본 cuda:0, 32B는 auto 권장).
    양자화 체크포인트는 transformers에서 조밀 텐서로 풀리므로 dtype을 bfloat16으로
    못 박는다. "auto"로 두면 fp32로 풀려 32B에서 메모리가 두 배가 된다.
    """
    from transformers import AutoModelForCausalLM
    dm = os.environ.get("QUANT_DEVICE_MAP", "cuda:0")
    kw = dict(trust_remote_code=True, device_map=dm, torch_dtype=torch.bfloat16)
    if is_bf16:
        kw.update(revision=REVISION)
    mm = os.environ.get("QUANT_MAX_MEMORY")   # 예: "0=40GiB,1=40GiB,cpu=80GiB"
    if mm:
        kw["max_memory"] = {
            (int(k.strip()) if k.strip().isdigit() else k.strip()): v.strip()
            for k, v in (x.split("=", 1) for x in mm.split(",") if x.strip())
        }
    m = AutoModelForCausalLM.from_pretrained(path_or_id, **kw)
    m.eval()
    return m'''

# ---------------------------------------------------------------- 5
# device_map="auto"로 모델이 여러 카드에 흩어지면 입력을 "cuda:0"에 고정하면 안 된다.
# 임베딩이 실제로 올라간 장치로 보내야 한다.
PE_OLD = '''                ids = tok(t, return_tensors="pt", truncation=True,
                          max_length=args.max_len).input_ids.to("cuda:0")'''

PE_NEW = '''                ids = tok(t, return_tensors="pt", truncation=True,
                          max_length=args.max_len).input_ids.to(_in_dev)'''

PE_OLD_2 = '''    @torch.no_grad()
    def per_item_nll(model):
        res = {}'''

PE_NEW_2 = '''    @torch.no_grad()
    def per_item_nll(model):
        # device_map="auto"에서는 임베딩이 올라간 장치로 입력을 보내야 한다.
        _in_dev = model.get_input_embeddings().weight.device
        res = {}'''


# 파일 -> (적용 완료 표지, [(old, new), ...])
# (그룹 이름, 파일, 적용 완료 표지, [(old, new), ...])
# 같은 파일에 독립된 그룹이 여러 개 있을 수 있고, 그룹별로 멱등하게 판단한다.
FIXES = [
    ("model-env", "activation_analysis.py", "QUANT_MODEL_ID",
     [(AA_OLD, AA_NEW)]),
    ("load-oom", "activation_analysis.py", "QUANT_DEVICE_MAP",
     [(AA2_OLD, AA2_NEW)]),
    ("ppl-device", "ppl_eval.py", "_in_dev",
     [(PE_OLD_2, PE_NEW_2), (PE_OLD, PE_NEW)]),
    ("rescore-ref", "rescore_all.py", "reported_from_results",
     [(RA_OLD_1, RA_NEW_1), (RA_OLD_2, RA_NEW_2),
      (RA_OLD_3, RA_NEW_3), (RA_OLD_4, RA_NEW_4)]),
    ("figure-cli", "make_figure1.py", "X_DEFAULT",
     [(MF_OLD_1, MF_NEW_1), (MF_OLD_2, MF_NEW_2),
      (MF_OLD_3, MF_NEW_3), (MF_OLD_4, MF_NEW_4),
      (MF_OLD_5, MF_NEW_5)]),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="실제로 적용한다")
    ap.add_argument("--revert", action="store_true", help=".bak 으로 되돌린다")
    ap.add_argument("--dir", default=".", help="대상 디렉터리")
    ap.add_argument("--only", nargs="*", help="특정 그룹만 처리 (예: load-oom ppl-device)")
    g = ap.parse_args()

    root = Path(g.dir)
    groups = [f for f in FIXES if not g.only or f[0] in g.only]
    if g.only:
        unknown = set(g.only) - {f[0] for f in FIXES}
        if unknown:
            print(f"알 수 없는 그룹: {sorted(unknown)}")
            print(f"가능: {[f[0] for f in FIXES]}")
            return 1
    problems = 0

    if g.revert:
        for name in sorted({f[1] for f in groups}):
            bak = root / (name + ".bak")
            if bak.exists():
                shutil.copy2(bak, root / name)
                print(f"되돌림 {name}  (<- {bak.name})")
            else:
                print(f"건너뜀 {name}  .bak 없음")
        return 0

    for gname, name, marker, pairs in groups:
        p = root / name
        label = f"{gname:12s} {name:24s}"
        if not p.exists():
            print(f"[없음]     {label}")
            problems += 1
            continue
        s = p.read_text(encoding="utf-8")

        if marker in s:
            print(f"[완료됨]   {label} ('{marker}' 발견 — 건드리지 않는다)")
            continue

        counts = [s.count(o) for o, _ in pairs]
        if any(c != 1 for c in counts):
            print(f"[불가]     {label} 앵커 매칭 횟수 {counts} (모두 1이어야 한다)")
            print(f"           서버 파일이 내 사본과 다르다. 해당 부분을 보여줄 것.")
            problems += 1
            continue

        if not g.write:
            print(f"[적용가능] {label} 수정 {len(pairs)}곳")
            continue

        new = s
        for o, n in pairs:
            new = new.replace(o, n)
        try:
            ast.parse(new)
        except SyntaxError as e:
            print(f"[실패]     {label} 적용 후 문법 오류: {e}")
            problems += 1
            continue
        bak = root / (name + ".bak")
        if not bak.exists():
            shutil.copy2(p, bak)
        p.write_text(new, encoding="utf-8")
        print(f"[적용]     {label} 수정 {len(pairs)}곳")

    if not g.write and problems == 0:
        print("\n실제 적용: python apply_fixes.py --write")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
