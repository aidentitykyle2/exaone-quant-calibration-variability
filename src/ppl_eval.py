#!/usr/bin/env python3
"""
ppl_eval.py — 퍼플렉시티를 네 번째 축으로 추가 (선행연구 비교용)

선행연구(Chimoto 등)는 퍼플렉시티를 주 지표로 썼다. 본 연구는 다운스트림 지표를
주 결과로 삼되, 같은 조건에서 PPL의 민감도가 어느 정도인지 보조 지표로 보고한다.
이로써 지표 비교 표가 네 축으로 완성된다.

  Accuracy           양자화 ↔ 정답
  Choice-KL          P_양자화 ↔ P_BF16 (선택지 제한)
  Decision Fidelity  판정 ↔ BF16 판정
  PPL                양자화 모델의 절대 언어모델링 성능 (BF16 대비 증가율로 환산)

측정 텍스트: KMMLU·MMLU 문항. 다른 세 지표와 같은 문항에서 재야 네 축을 직접 비교할
수 있고, 캘리브레이션에 쓰이지 않은 보류 텍스트라는 조건도 만족한다.
한계: 일반적인 PPL 측정은 긴 자연문 말뭉치를 쓴다. 객관식 문항 텍스트로 잰 값이므로
      절대값을 선행연구와 직접 비교하지 말고, 조건 간 상대 비교에만 사용한다.

두 단계로 나뉘어 있어 GPU별 병렬 실행이 가능하다.
  measure   모델별 문항 단위 NLL을 저장
  aggregate 저장된 값으로 PPL 증가율과 필요 시드 수를 계산

사용법:
  # 4장에 나눠 측정 (조건·시드 분할)
  CUDA_VISIBLE_DEVICES=0 python ppl_eval.py measure --base $B --conds ko --seeds 0 1 2 3 4 5
  CUDA_VISIBLE_DEVICES=1 python ppl_eval.py measure --base $B --conds ko --seeds 6 7 8 9 10
  CUDA_VISIBLE_DEVICES=2 python ppl_eval.py measure --base $B --conds en_instruct --seeds 0 1 2 3 4 5
  CUDA_VISIBLE_DEVICES=3 python ppl_eval.py measure --base $B --conds en_instruct --seeds 6 7 8 9 10
  # BF16은 한 번만 (아무 GPU에서)
  CUDA_VISIBLE_DEVICES=0 python ppl_eval.py measure --base $B --bf16-only

  python ppl_eval.py aggregate --base $B
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))


def measure(args):
    import torch
    from activation_analysis import MODEL_ID, REVISION, free, heldout_texts, load_model
    from transformers import AutoTokenizer

    base = Path(args.base)
    out = base / "ppl"
    out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION, trust_remote_code=True)
    texts = {"ko": heldout_texts("ko", args.per_subject),
             "en": heldout_texts("en", args.per_subject)}
    print(f"문항 수: 한국어 {len(texts['ko'])}, 영어 {len(texts['en'])}", flush=True)

    @torch.no_grad()
    def per_item_nll(model):
        # device_map="auto"에서는 임베딩이 올라간 장치로 입력을 보내야 한다.
        _in_dev = model.get_input_embeddings().weight.device
        res = {}
        for lang, ts in texts.items():
            rec = []
            for t in ts:
                ids = tok(t, return_tensors="pt", truncation=True,
                          max_length=args.max_len).input_ids.to(_in_dev)
                if ids.shape[1] < 2:
                    continue
                logits = model(ids).logits[0, :-1].float()
                lp = torch.log_softmax(logits, -1)
                tgt = ids[0, 1:]
                nll = -lp.gather(1, tgt[:, None]).squeeze(1).sum().item()
                rec.append((nll, int(tgt.numel())))
            res[lang] = rec
        return res

    targets = [("bf16", MODEL_ID, True)]
    if not args.bf16_only:
        targets = []
        for c in args.conds:
            for s in args.seeds:
                name = "gptq_4bit" + ("" if c == "ko" else f"_calib_{c}") + ("" if s == 0 else f"_s{s}")
                targets.append((f"gptq_{c}_s{s}", str(base / name), False))

    for label, path, is_bf16 in targets:
        f = out / f"{label}.json"
        if f.exists() and not args.overwrite:
            print(f"건너뜀 {label} (이미 있음)", flush=True)
            continue
        if not is_bf16 and not Path(path).is_dir():
            print(f"건너뜀 {label} (체크포인트 없음)", flush=True)
            continue
        print(f"측정 {label}", flush=True)
        m = load_model(path, is_bf16=is_bf16)
        f.write_text(json.dumps(per_item_nll(m)), encoding="utf-8")
        free(m)
    print(f"저장 → {out}")


def aggregate(args):
    base = Path(args.base)
    d = base / "ppl"
    rng = np.random.default_rng(11)
    bf = json.loads((d / "bf16.json").read_text(encoding="utf-8"))
    models = {}
    for f in sorted(d.glob("gptq_*.json")):
        c, s = f.stem[5:].rsplit("_s", 1)
        models[(c, int(s))] = json.loads(f.read_text(encoding="utf-8"))
    if not models:
        print("측정 결과가 없다. measure 단계를 먼저 실행할 것.")
        return

    lines = ["=== PPL (BF16 대비 증가율, %) ===",
             "    측정 텍스트: KMMLU·MMLU 문항. 다른 지표와 같은 문항에서 측정했다.",
             "    절대 PPL은 객관식 문항 기준이므로 선행연구와 직접 비교하지 않는다.", ""]

    def ppl(rec):
        n = np.array([x[0] for x in rec])
        t = np.array([x[1] for x in rec])
        return float(np.exp(n.sum() / t.sum()))

    conds = sorted({c for c, _ in models})
    for lang, bench in (("ko", "KMMLU"), ("en", "MMLU")):
        base_ppl = ppl(bf[lang])
        lines.append(f"  {bench} BF16 PPL = {base_ppl:.3f}")
        for c in conds:
            seeds = sorted(s for cc, s in models if cc == c)
            inc = [(ppl(models[(c, s)][lang]) / base_ppl - 1) * 100 for s in seeds]
            lines.append(f"    gptq[{c:11s}] 시드 {len(seeds):2d}개  증가율 평균 {np.mean(inc):+6.2f}%  "
                         f"SD {np.std(inc, ddof=1):.2f}  범위 [{min(inc):+.2f}, {max(inc):+.2f}]")
        lines.append("")

    # 필요 시드 수: 다른 지표와 같은 2단계 재표집 절차
    lines.append("=== 필요 시드 수 (2단계 부트스트랩, 다른 지표와 동일 절차) ===")
    for lang, bench, worse, better in (("ko", "KMMLU", "en_instruct", "ko"),
                                       ("en", "MMLU", "ko", "en_instruct")):
        if worse not in conds or better not in conds:
            continue
        base_n = np.array([x[0] for x in bf[lang]])
        base_t = np.array([x[1] for x in bf[lang]])
        cols = {}
        for c in (worse, better):
            seeds = sorted(s for cc, s in models if cc == c)
            cols[c] = (seeds,
                       np.column_stack([[x[0] for x in models[(c, s)][lang]] for s in seeds]),
                       np.column_stack([[x[1] for x in models[(c, s)][lang]] for s in seeds]))
        n_items = len(base_n)
        found = None
        med = None
        for k in range(2, 41):
            stat = np.empty(args.iters)
            for i in range(args.iters):
                w = rng.multinomial(n_items, np.full(n_items, 1.0 / n_items))
                bp = np.exp((w @ base_n) / (w @ base_t))
                vals = []
                for c in (worse, better):
                    seeds, N, T = cols[c]
                    sel = rng.integers(0, len(seeds), k)
                    p = np.exp((w @ N[:, sel]) / (w @ T[:, sel]))
                    vals.append(np.mean((p / bp - 1) * 100))
                stat[i] = vals[0] - vals[1]
            lo, hi = np.quantile(stat, [0.025, 0.975])
            if med is None:
                med = float(np.median(stat))
            if lo > 0 or hi < 0:
                found = k
                break
        lines.append(f"  {bench:6s} PPL 증가율 차이 {med:+.3f}%  필요 시드 "
                     f"{found if found else '>40'}  (보유 {len(cols[better][0])})")

    text = "\n".join(lines)
    (d / "ppl_summary.txt").write_text(text + "\n", encoding="utf-8")
    print(text)
    print(f"\n저장 → {d}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("measure")
    m.add_argument("--base", required=True)
    m.add_argument("--conds", nargs="*", default=["ko", "en_instruct"])
    m.add_argument("--seeds", nargs="*", type=int, default=list(range(11)))
    m.add_argument("--per-subject", type=int, default=16)
    m.add_argument("--max-len", type=int, default=768)
    m.add_argument("--bf16-only", action="store_true")
    m.add_argument("--overwrite", action="store_true")
    m.set_defaults(func=measure)
    a = sub.add_parser("aggregate")
    a.add_argument("--base", required=True)
    a.add_argument("--iters", type=int, default=300)
    a.set_defaults(func=aggregate)
    args = ap.parse_args()
    args.func(args)
