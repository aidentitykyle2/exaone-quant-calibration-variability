#!/usr/bin/env python3
"""
rescore_all.py — seed_eval/ 아래 모든 평가 기록을 같은 채점 기준으로 다시 집계

논문의 모든 수치를 한 가지 채점 기준으로 통일하기 위한 스크립트다.
기존 표는 lm-eval의 첫 번째 선택 규칙(동률이면 A)으로 채점되어 있었고,
이는 bf16 로짓 동률 때문에 조건마다 다르게 편향되었다.

채점 규칙
  acc 계열 과제 (KMMLU, MMLU, KoBEST BoolQ/COPA/WiC/SentiNeg, BoolQ, HAE-RAE)
    선택지 로그우도로 직접 채점한다. first(동률이면 첫 번째)와 tie_split(동률 분할)을
    모두 계산한다. HAE-RAE는 선택지가 모두 한 글자라 acc_norm과 acc가 같다.
  acc_norm 계열 과제 (KoBEST HellaSwag, HellaSwag, ARC-Challenge)
    선택지가 여러 토큰이라 원시 로그우도 동률은 드물다. 길이 정규화를 재현하려면
    선택지 문자열 길이가 필요하므로, 기록에 저장된 acc_norm을 그대로 쓴다
    (두 기준에 같은 값). 원시 로그우도 동률률은 참고로 보고한다.

그룹 과제(KMMLU, MMLU, HAE-RAE)는 하위 과제 문항을 모두 모아 문항 평균(micro)을 쓴다.
정합성 점검 절에서 first 기준값이 기존 lm-eval 보고치와 일치하는지 대조한다.

사용법:
  python rescore_all.py --base quant_experiments/EXAONE-3.5-7.8B-Instruct
"""

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

SAMPLE_RE = re.compile(r"samples_(.+?)_(\d{4}-\d{2}-\d{2}T[\d\-.]+)\.jsonl$")
ACC_NORM_TASKS = {"kobest_hellaswag", "hellaswag", "arc_challenge"}


def _continuations(r):
    """기록의 arguments에서 선택지별 이어쓰기 문자열을 순서대로 꺼낸다.
    lm-eval 버전에 따라 [[문맥, 이어쓰기], ...] 리스트이거나
    {"gen_args_0": {"arg_0": 문맥, "arg_1": 이어쓰기}, ...} 딕셔너리다."""
    args = r.get("arguments")
    if isinstance(args, dict):
        keys = sorted(args, key=lambda k: int(re.sub(r"\D", "", k) or 0))
        seq = [args[k] for k in keys]
    elif isinstance(args, list):
        seq = args
    else:
        return None
    out = []
    for e in seq:
        if isinstance(e, dict):
            out.append(e.get("arg_1"))
        elif isinstance(e, (list, tuple)) and len(e) >= 2:
            out.append(e[1])
        else:
            return None
    return out


def target_index(r):
    """정답을 선택지 번호로 바꾼다. 정수이면 그대로, 문자열(예: HAE-RAE의 '(A)')이면
    선택지 이어쓰기 문자열과 대조해 위치를 찾는다. 찾지 못하면 None."""
    t = r.get("target")
    if isinstance(t, int):
        return t
    try:
        return int(t)
    except (TypeError, ValueError):
        pass
    conts = _continuations(r)
    if conts:
        ts = str(t).strip()
        for i, cont in enumerate(conts):
            if cont is not None and str(cont).strip() == ts:
                return i
    return None

# 표에 쓸 과제: (표시 이름, 매칭 방식, 매칭 값)
TASKS = {
    "ko": [("kmmlu", "prefix", "kmmlu_"), ("kobest_hellaswag", "exact", "kobest_hellaswag"),
           ("kobest_boolq", "exact", "kobest_boolq"), ("kobest_copa", "exact", "kobest_copa"),
           ("kobest_wic", "exact", "kobest_wic"), ("kobest_sentineg", "exact", "kobest_sentineg"),
           ("haerae", "prefix", "haerae_")],
    "en": [("mmlu", "prefix", "mmlu_"), ("hellaswag", "exact", "hellaswag"),
           ("boolq", "exact", "boolq"), ("arc_challenge", "exact", "arc_challenge")],
}
PAIRS = [("MMLU계열", "kmmlu", "mmlu"), ("HellaSwag", "kobest_hellaswag", "hellaswag"),
         ("BoolQ", "kobest_boolq", "boolq")]
# 정합성 점검용 참조값.
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
    return out


def match(task, how, val):
    return task.startswith(val) if how == "prefix" else task == val


def score_dir(model_dir: Path, lang: str):
    """과제 표시이름 -> dict(n, first, tie_split, ties)"""
    latest = {}
    for p in model_dir.rglob("samples_*.jsonl"):
        m = SAMPLE_RE.search(p.name)
        if m and (m.group(1) not in latest or m.group(2) > latest[m.group(1)][0]):
            latest[m.group(1)] = (m.group(2), p)

    agg = {name: defaultdict(float) for name, _, _ in TASKS[lang]}
    mismatch = defaultdict(int)
    for task, (_, p) in latest.items():
        target_name = next((n for n, how, v in TASKS[lang] if match(task, how, v)), None)
        if target_name is None:
            continue
        a = agg[target_name]
        with open(p, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                fr = r.get("filtered_resps")
                if not fr:
                    continue
                ll = [float(x[0]) if isinstance(x, (list, tuple)) else float(x) for x in fr]
                mx = max(ll)
                tied = [i for i, v in enumerate(ll) if v == mx]
                a["n"] += 1
                a["ties"] += len(tied) > 1
                if target_name in ACC_NORM_TASKS:
                    v = float(r.get("acc_norm", r.get("acc", 0.0)))
                    a["first"] += v
                    a["tie_split"] += v
                else:
                    t = target_index(r)
                    if t is None:
                        a["n"] -= 1
                        a["ties"] -= len(tied) > 1
                        mismatch[target_name + "(정답 해석 실패)"] += 1
                        continue
                    first = float(ll.index(mx) == t)
                    a["first"] += first
                    a["tie_split"] += (1.0 if t in tied else 0.0) / len(tied)
                    if "acc" in r and float(r["acc"]) != first:
                        mismatch[target_name] += 1
    res = {}
    for name, a in agg.items():
        if a["n"]:
            res[name] = {"n": int(a["n"]), "first": a["first"] / a["n"],
                         "tie_split": a["tie_split"] / a["n"], "ties": a["ties"] / a["n"]}
    return res, dict(mismatch)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    args = ap.parse_args()
    base = Path(args.base)
    out = base / "rescore"
    out.mkdir(parents=True, exist_ok=True)

    data = {}       # (lang, model) -> 결과
    reported = {}   # (lang, model) -> lm-eval이 그 실행에서 기록한 정확도
    lines = []
    for lang in ("ko", "en"):
        root = base / "seed_eval" / lang
        if not root.is_dir():
            continue
        for d in sorted(p for p in root.iterdir() if p.is_dir()):
            res, mism = score_dir(d, lang)
            if res:
                data[(lang, d.name)] = res
                reported[(lang, d.name)] = reported_from_results(
                    d, [n for n, _, _ in TASKS[lang]])
                bad = {k: v for k, v in mism.items() if "해석 실패" not in k}
                unresolved = {k: v for k, v in mism.items() if "해석 실패" in k}
                if bad:
                    lines.append(f"경고: {lang}/{d.name} 기록된 acc와 재계산 점수가 다른 문항 {bad} — 채점 재현 오류")
                if unresolved:
                    lines.append(f"참고: {lang}/{d.name} 정답을 선택지에서 찾지 못해 제외한 문항 {unresolved}")

    models = sorted({m for _, m in data}, key=lambda x: (x != "bf16", x))

    lines.append("=== 정합성 점검: first 기준이 그 실행의 lm-eval 보고치와 일치하는가 ===")
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
        lines.append(f"    대조한 (조건, 과제) 쌍 {n_checked}개. 불일치만 위에 표시(bf16은 항상 표시).")
    lines.append("")

    rows = []
    for scoring in ("first", "tie_split"):
        for lang in ("ko", "en"):
            names = [n for n, _, _ in TASKS[lang]]
            bf = data.get((lang, "bf16"))
            lines.append(f"=== [{scoring}] {'한국어' if lang == 'ko' else '영어'} 원점수 ===")
            lines.append(f"  {'모델':24s}" + "".join(f"{n[:13]:>14s}" for n in names))
            for m in models:
                r = data.get((lang, m))
                if r:
                    lines.append(f"  {m:24s}" + "".join(
                        f"{r[n][scoring]:14.4f}" if n in r else f"{'-':>14s}" for n in names))
                    for n in names:
                        if n in r:
                            deg = ((bf[n][scoring] - r[n][scoring]) / bf[n][scoring] * 100
                                   if bf and n in bf else None)
                            rows.append([scoring, lang, m, n, round(r[n][scoring], 4),
                                         None if deg is None else round(deg, 2), round(r[n]["ties"], 4)])
            if bf:
                lines.append(f"=== [{scoring}] {'한국어' if lang == 'ko' else '영어'} BF16 대비 저하율 (%) ===")
                for m in models:
                    r = data.get((lang, m))
                    if r and m != "bf16":
                        lines.append(f"  {m:24s}" + "".join(
                            f"{(bf[n][scoring] - r[n][scoring]) / bf[n][scoring] * 100:14.2f}"
                            if n in r and n in bf else f"{'-':>14s}" for n in names))
            lines.append("")

        lines.append(f"=== [{scoring}] 언어별 열화 비대칭 (한국어 저하율 − 영어 저하율, %p) ===")
        bk, be = data.get(("ko", "bf16")), data.get(("en", "bf16"))
        for m in models:
            rk, re_ = data.get(("ko", m)), data.get(("en", m))
            if m == "bf16" or not (rk and re_ and bk and be):
                continue
            cells = []
            for lab, tk, te in PAIRS:
                if tk in rk and te in re_:
                    dk = (bk[tk][scoring] - rk[tk][scoring]) / bk[tk][scoring] * 100
                    de = (be[te][scoring] - re_[te][scoring]) / be[te][scoring] * 100
                    cells.append(f"{lab}={dk - de:+.2f}")
            lines.append(f"  {m:24s} " + "  ".join(cells))
        lines.append("")

    lines.append("=== 동률률 (원시 로그우도 기준) ===")
    for lang in ("ko", "en"):
        names = [n for n, _, _ in TASKS[lang]]
        lines.append(f"  [{lang}] {'모델':20s}" + "".join(f"{n[:13]:>14s}" for n in names))
        for m in models:
            r = data.get((lang, m))
            if r:
                lines.append(f"       {m:20s}" + "".join(
                    f"{r[n]['ties'] * 100:13.1f}%" if n in r else f"{'-':>14s}" for n in names))
    lines.append("  (acc_norm 과제의 동률률은 참고용이며 채점에는 기록된 acc_norm을 사용)")

    text = "\n".join(lines)
    (out / "rescore_summary.txt").write_text(text + "\n", encoding="utf-8")
    with open(out / "rescore_long.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["scoring", "lang", "model", "task", "score", "degradation_pct", "tie_rate"])
        w.writerows(rows)
    print(text)
    print(f"\n저장 → {out}")


if __name__ == "__main__":
    main()
