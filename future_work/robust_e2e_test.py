#!/usr/bin/env python3
"""
robust_e2e_test.py — RobustGPTQModifier의 실제 파이프라인 통합 검증 (CPU, 수 분)

*** 향후 연구용 코드다. 본 논문의 결과에는 사용되지 않았다. ***

robust_selftest.py는 집계 수식만 본다. 이 스크립트는 llmcompressor의 실제
oneshot 경로를 타고, 무작위 초기화한 아주 작은 Llama에 대해

    (A) 기준선 GPTQModifier
    (B) RobustGPTQModifier(K=4, aggregation="mean")

두 결과의 양자화 가중치가 일치하는지 확인한다. mean 집계는 기존 GPTQ와
수학적으로 동일하므로 일치해야 하며, 일치하지 않으면 훅·부분집합 배정·
QAC 비활성화 등 **파이프라인 배선**에 문제가 있다는 뜻이다.

이 검증은 GPU도 실제 모델 다운로드도 필요 없다(모델을 난수로 만든다).
따라서 서버 시간을 쓰기 전에 노트북에서도 돌릴 수 있다. 실제 대상 모델에서의
검증은 robust_job.sh 의 rcheck 작업이 담당한다.

사용법:
  python robust_e2e_test.py
  python robust_e2e_test.py --subsets 4 --samples 32 --layers 2
"""

import argparse
import sys

import torch


def build_tiny_model(layers: int, hidden: int, inter: int, vocab: int):
    from transformers import LlamaConfig, LlamaForCausalLM

    cfg = LlamaConfig(
        vocab_size=vocab, hidden_size=hidden, intermediate_size=inter,
        num_hidden_layers=layers, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=256, tie_word_embeddings=False,
    )
    torch.manual_seed(0)
    return LlamaForCausalLM(cfg).to(torch.float32).eval()


def build_tiny_tokenizer(vocab: int):
    """
    oneshot()은 dataset을 받을 때 processor를 요구한다. 데이터셋이 이미
    input_ids를 담고 있어 실제 토큰화는 일어나지 않으므로, 배선만 만족시키는
    최소 토크나이저를 오프라인으로 만든다(모델 다운로드 불필요).
    """
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    t = Tokenizer(models.WordLevel(
        vocab={f"t{i}": i for i in range(vocab)}, unk_token="t0"))
    t.pre_tokenizer = pre_tokenizers.Whitespace()
    return PreTrainedTokenizerFast(tokenizer_object=t, unk_token="t0",
                                   pad_token="t0", model_max_length=vocab)


def make_dataset(n: int, seq: int, vocab: int):
    from datasets import Dataset

    g = torch.Generator().manual_seed(1234)
    ids = torch.randint(0, vocab, (n, seq), generator=g)
    return Dataset.from_dict({
        "input_ids": ids.tolist(),
        "attention_mask": torch.ones_like(ids).tolist(),
    })


def quantized_params(model) -> dict:
    """양자화 후 각 Linear의 가중치·스케일·영점을 모은다."""
    out = {}
    for name, mod in model.named_modules():
        if not isinstance(mod, torch.nn.Linear):
            continue
        for attr in ("weight", "weight_scale", "weight_zero_point", "weight_g_idx"):
            t = getattr(mod, attr, None)
            if t is not None:
                out[f"{name}.{attr}"] = t.detach().to(torch.float64).clone()
    return out


def run(recipe, ds, tok, args):
    from llmcompressor import oneshot

    model = build_tiny_model(args.layers, args.hidden, args.inter, args.vocab)
    oneshot(model=model, processor=tok, dataset=ds, recipe=recipe,
            max_seq_length=args.seq, num_calibration_samples=args.samples,
            pipeline="sequential")
    return quantized_params(model)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subsets", type=int, default=4)
    ap.add_argument("--samples", type=int, default=32)
    ap.add_argument("--seq", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--inter", type=int, default=256)
    ap.add_argument("--vocab", type=int, default=512)
    ap.add_argument("--tol", type=float, default=1e-4,
                    help="가중치 최대 상대오차 허용값. 부동소수점 누적 순서 차이만 남아야 한다.")
    a = ap.parse_args()

    from llmcompressor.modifiers.quantization import GPTQModifier

    from robust_gptq import RobustGPTQModifier

    ds = make_dataset(a.samples, a.seq, a.vocab)
    tok = build_tiny_tokenizer(a.vocab)

    print(f"설정: K={a.subsets} samples={a.samples} seq={a.seq} "
          f"layers={a.layers} hidden={a.hidden} inter={a.inter}")
    print("-" * 62)

    print("[A] 기준선 GPTQModifier 실행...")
    ref = run([GPTQModifier(targets="Linear", scheme="W4A16", ignore=["lm_head"])], ds, tok, a)

    print("[B] RobustGPTQModifier(K=%d, agg=mean) 실행..." % a.subsets)
    got = run([RobustGPTQModifier(
        targets="Linear", scheme="W4A16", ignore=["lm_head"],
        num_subsets=a.subsets, aggregation="mean",
        num_calibration_batches=a.samples)], ds, tok, a)

    print("-" * 62)
    only_a = sorted(set(ref) - set(got))
    only_b = sorted(set(got) - set(ref))
    if only_a or only_b:
        print(f"파라미터 구성이 다르다. 기준선만: {only_a[:5]} / 비교만: {only_b[:5]}")
        return 1

    worst, worst_name = 0.0, "(모두 동일)"
    for k in sorted(ref):
        x, y = ref[k], got[k]
        if x.shape != y.shape:
            print(f"형상 불일치: {k} {tuple(x.shape)} vs {tuple(y.shape)}")
            return 1
        denom = max(float(x.abs().max()), 1e-30)
        e = float((x - y).abs().max()) / denom
        if e > worst:
            worst, worst_name = e, k
    print(f"비교 파라미터 {len(ref)}개 | 최대 상대오차 {worst:.3e} ({worst_name})")

    if worst <= a.tol:
        print("판정: 일치 — 부분집합 분할·훅·집계 배선이 기준선과 동일하다.")
        print("      다음은 실제 대상 모델에서의 검증이다: robust_job.sh rcheck:ko:1:4")
        return 0

    print("판정: 불일치 — 배선에 문제가 있다. 확인 순서:")
    print("  1) DISABLE_QAC_MODIFIERS에 RobustGPTQModifier가 등록되었는지")
    print("  2) 로그의 '부분집합 표본 수' 합이 총 표본 수와 같은지(중복·누락 여부)")
    print("  3) subset_assignment가 block인데 총 배치 수를 잘못 받았는지")
    return 1


if __name__ == "__main__":
    sys.exit(main())
