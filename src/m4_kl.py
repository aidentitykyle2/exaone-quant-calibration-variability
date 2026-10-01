#!/usr/bin/env python3
"""
m4_kl.py — 출력 분포 수준의 양자화 충실도 (M4)

M3(은닉 상태 상대 오차)는 GPTQ에서는 정확도 교차와 일치했으나, AWQ 한국어에서는
최종층에서 방향이 반대로 나타났다. 은닉 상태 오차는 4,096차원 전체를 동일 가중으로
합산하므로, 정답 판정에 쓰이지 않는 차원(예: 최종층의 거대 활성화 차원)의 오차가
지표를 지배할 수 있다. M4는 정답 판정이 실제로 일어나는 위치에서 출력 분포를
직접 비교해 이 문제를 피한다.

측정 위치: 문항 끝("정답:" / "Answer:") 바로 다음 토큰의 예측 분포
측정량 (BF16 대비):
  kl_full    전체 어휘에 대한 KL(P_BF16 || P_quant)
  kl_choice  A·B·C·D 네 토큰으로 제한·재정규화한 분포의 KL  ← 정답 판정과 직결
  flip       네 토큰 중 최대 확률 선택지가 BF16과 달라진 비율  ← 가장 직관적인 지표

교차 판정 (M3과 같은 방향 정의, 양수가 예상 방향):
  Δ_KO = 지표(en_instruct 캘리브, 한국어) − 지표(ko 캘리브, 한국어)
  Δ_EN = 지표(ko 캘리브, 영어)           − 지표(en_instruct 캘리브, 영어)

유의성: 같은 문항에 대해 두 모델을 비교하므로 문항 단위 대응 부트스트랩을 쓴다.

한계: 0-shot 조건이며 lm-eval의 5-shot 로그우도 채점과 완전히 같지는 않다.
      절대 정확도가 아니라 BF16 대비 충실도를 두 캘리브레이션 조건 간에 비교하는
      목적에는 충분하다.

사용법:
  CUDA_VISIBLE_DEVICES=0 python m4_kl.py \\
      --base quant_experiments/EXAONE-3.5-7.8B-Instruct \\
      --out  quant_experiments/EXAONE-3.5-7.8B-Instruct/mechanism
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from activation_analysis import MODEL_ID, REVISION, free, heldout_texts, load_model  # noqa: E402

LETTERS = ["A", "B", "C", "D"]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def letter_ids(tok):
    """선택지 문자 토큰 ID. ' A'가 단일 토큰이면 그것을, 아니면 'A'를 쓴다."""
    ids = []
    for L in LETTERS:
        chosen = None
        for cand in (" " + L, L):
            enc = tok.encode(cand, add_special_tokens=False)
            if len(enc) == 1:
                chosen = enc[0]
                break
        if chosen is None:
            raise RuntimeError(f"선택지 '{L}'가 단일 토큰으로 인코딩되지 않는다.")
        ids.append(chosen)
    if len(set(ids)) != 4:
        raise RuntimeError(f"선택지 토큰 ID가 중복된다: {ids}")
    return ids


@torch.no_grad()
def last_logprobs(model, tok, text, max_len):
    ids = tok(text, return_tensors="pt", truncation=True, max_length=max_len).input_ids.to("cuda:0")
    logits = model(ids).logits[0, -1].float()
    return torch.log_softmax(logits, dim=-1).cpu()


def kl(logp, logq):
    p = logp.exp()
    return float((p * (logp - logq)).sum())


def choice_logprobs(full_logp, ids):
    sub = full_logp[ids]
    return sub - torch.logsumexp(sub, dim=0)


def paired_boot(delta, iters, rng):
    delta = np.asarray(delta, dtype=np.float64)
    n = len(delta)
    means = np.array([delta[rng.integers(0, n, n)].mean() for _ in range(iters)])
    return float(delta.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975)), \
        float((means <= 0).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-subject", type=int, default=16, help="과목당 문항 수 (8과목 × 언어)")
    ap.add_argument("--max-len", type=int, default=768)
    ap.add_argument("--iters", type=int, default=2000)
    args = ap.parse_args()

    base, out = Path(args.base), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(12345)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION, trust_remote_code=True)
    # 문항 끝(정답 위치)이 잘리지 않도록 앞쪽을 자른다.
    tok.truncation_side = "left"
    ids = letter_ids(tok)
    log(f"선택지 토큰 ID: {dict(zip(LETTERS, ids))}")

    texts = {"ko": heldout_texts("ko", args.per_subject), "en": heldout_texts("en", args.per_subject)}
    log(f"문항 수: 한국어 {len(texts['ko'])}, 영어 {len(texts['en'])}")
    if not texts["ko"] or not texts["en"]:
        log("오류: 문항을 확보하지 못했다.")
        sys.exit(1)

    log("BF16 기준 분포 계산")
    ref = load_model(MODEL_ID, is_bf16=True)
    ref_lp = {lang: [last_logprobs(ref, tok, t, args.max_len) for t in ts] for lang, ts in texts.items()}
    free(ref)

    ckpts = {
        "gptq_ko": base / "gptq_4bit",
        "gptq_en_instruct": base / "gptq_4bit_calib_en_instruct",
        "awq_ko": base / "awq_4bit",
        "awq_en_instruct": base / "awq_4bit_calib_en_instruct",
    }
    res = {}   # (model, lang) -> dict(metric -> np.array)
    rows = []
    for name, path in ckpts.items():
        if not path.is_dir():
            log(f"{name}: 체크포인트 없음, 건너뜀")
            continue
        log(f"{name} 로딩")
        q = load_model(str(path), is_bf16=False)
        for lang, ts in texts.items():
            kf, kc, fl = [], [], []
            for i, t in enumerate(ts):
                lq = last_logprobs(q, tok, t, args.max_len)
                lp = ref_lp[lang][i]
                cp, cq = choice_logprobs(lp, ids), choice_logprobs(lq, ids)
                kf.append(kl(lp, lq))
                kc.append(kl(cp, cq))
                fl.append(float(int(cp.argmax()) != int(cq.argmax())))
                rows.append([name, lang, i, kf[-1], kc[-1], fl[-1]])
            res[(name, lang)] = {"kl_full": np.array(kf), "kl_choice": np.array(kc), "flip": np.array(fl)}
            log(f"  {name} × {lang}: kl_full={np.mean(kf):.4f}  kl_choice={np.mean(kc):.4f}  "
                f"flip={np.mean(fl):.3f}")
        free(q)

    with open(out / "m4_output_kl.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["model", "eval_lang", "item", "kl_full", "kl_choice", "flip"])
        w.writerows(rows)

    lines = ["=== M4 출력 분포 충실도 (정답 위치, BF16 대비) ===",
             "    Δ_KO = 지표(en_instruct 캘리브, 한국어) − 지표(ko 캘리브, 한국어)  → 양수 예상",
             "    Δ_EN = 지표(ko 캘리브, 영어) − 지표(en_instruct 캘리브, 영어)      → 양수 예상",
             f"    문항 단위 대응 부트스트랩 {args.iters}회, 95% 구간",
             ""]
    lines.append("--- 조건별 평균 ---")
    for (name, lang), m in res.items():
        lines.append(f"  {name:18s} {lang}  kl_full={m['kl_full'].mean():.4f}  "
                     f"kl_choice={m['kl_choice'].mean():.4f}  flip={m['flip'].mean():.3f}")
    lines.append("")
    lines.append("--- 교차 판정 ---")
    for meth in ("gptq", "awq"):
        k, e = f"{meth}_ko", f"{meth}_en_instruct"
        if not all((x, l) in res for x in (k, e) for l in ("ko", "en")):
            lines.append(f"  {meth.upper()}: 체크포인트 누락으로 계산 불가")
            continue
        for metric in ("kl_full", "kl_choice", "flip"):
            d_ko = res[(e, "ko")][metric] - res[(k, "ko")][metric]
            d_en = res[(k, "en")][metric] - res[(e, "en")][metric]
            mk, lk, hk, pk = paired_boot(d_ko, args.iters, rng)
            me, le, he, pe = paired_boot(d_en, args.iters, rng)
            tag = lambda lo, hi: "0 미포함" if (lo > 0 or hi < 0) else "0 포함"
            lines.append(f"  {meth.upper():4s} {metric:9s} Δ_KO={mk:+.4f} [{lk:+.4f},{hk:+.4f}] {tag(lk, hk)}"
                         f"   Δ_EN={me:+.4f} [{le:+.4f},{he:+.4f}] {tag(le, he)}")
    lines.append("")
    lines.append("해석: AWQ의 Δ_KO가 kl_choice·flip에서 양수면, M3의 최종층 역전은 정답 판정과 무관한")
    lines.append("      차원의 오차였다는 뜻이며 교차는 출력 수준에서 두 기법 모두 성립한다.")

    text = "\n".join(lines)
    (out / "m4_summary.txt").write_text(text + "\n", encoding="utf-8")
    print("\n" + text)
    log(f"저장 → {out}")


if __name__ == "__main__":
    main()
