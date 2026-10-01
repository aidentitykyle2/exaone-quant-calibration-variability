#!/usr/bin/env python3
"""
activation_analysis.py — 캘리브레이션 언어 효과의 기제 분석

정확도 교차(한국어 캘리브레이션 → 한국어 보존, 영어 캘리브레이션 → 영어 보존)가
왜 발생하는지를 세 가지 측정으로 검증한다.

  M1. 중요 채널 중첩도 (AWQ 기제)
      레이어별 선형층 입력에서 평균 |활성화| 상위 1% 채널을 언어별로 구하고
      Jaccard 지수를 계산한다. AWQ는 이 채널들을 보호 대상으로 판정한다.

  M2. 헤시안 대각 유사도 (GPTQ 기제)
      레이어별 입력의 채널별 평균 제곱(= XᵀX 대각/n)을 구하고
      언어 간 순위 상관(Spearman)을 계산한다. GPTQ의 오차 보정 가중치가 된다.

  M3. 언어별 레이어 출력 오차 (기제와 정확도 결과의 연결고리)
      양자화 모델에 보류(held-out) 한국어·영어 텍스트를 넣어
      레이어별 은닉 상태의 BF16 대비 상대 오차를 측정한다.

설계상 핵심: 언어 내 기준선(noise floor)
  M1·M2의 "한국어 vs 영어" 값은 단독으로는 해석할 수 없다. 같은 언어의 서로 다른
  표본 묶음(seed 0 vs seed 1)끼리의 값을 함께 측정해, 언어 간 차이가 표본 변동을
  넘어서는지 판단한다.

  ko(s0) vs ko(s1)            → 한국어 내 변동 (기준선)
  en_instruct(s0) vs (s1)     → 영어 내 변동 (기준선)
  ko(s0) vs en_instruct(s0)   → 언어 간 차이 (검증 대상)

  en_instruct를 쓰는 이유: 도메인(instruction)을 고정한 순수 언어 비교이기 때문이다.

M3의 보류 텍스트
  캘리브레이션에 쓴 텍스트로 측정하면 순환 논증이 된다. 정확도 평가에 쓴
  KMMLU·MMLU 문항을 사용한다. 캘리브레이션에 쓰이지 않았고, 이미 캐시되어 있어
  오프라인으로 로드되며, 정확도 교차가 관찰된 바로 그 분포이므로 기제와 결과를
  직접 연결할 수 있다.

M3은 AWQ·GPTQ에만 적용한다. QuaRot은 R1 회전으로 은닉 상태의 좌표계 자체가
바뀌므로 BF16 은닉 상태와 직접 비교할 수 없다.

사용법:
  CUDA_VISIBLE_DEVICES=0 python activation_analysis.py \\
      --base quant_experiments/EXAONE-3.5-7.8B-Instruct \\
      --out  quant_experiments/EXAONE-3.5-7.8B-Instruct/mechanism

  모델 로딩만 빠르게 점검하려면 --smoke
"""

import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")  # 캐시된 데이터만 사용(404 재발 방지)

import argparse
import csv
import gc
import re
import sys
import time
from pathlib import Path

import torch

# 모델을 환경변수로 덮어쓸 수 있게 한다. ppl_eval.py가 이 두 값을 import해
# BF16 기준 모델과 토크나이저를 정하므로, 하드코딩된 상태에서 32B를 측정하면
# 양자화 체크포인트는 32B인데 BF16 기준만 7.8B가 되어 PPL 증가율이 무의미해진다.
#   QUANT_MODEL_ID=LGAI-EXAONE/EXAONE-3.5-32B-Instruct \
#   QUANT_REVISION=d6fa88cd8d2c9512b40578bdc44e64909e5a5042 python ppl_eval.py measure ...
# QUANT_REVISION을 빈 문자열로 두면 리비전을 고정하지 않는다.
MODEL_ID = os.environ.get("QUANT_MODEL_ID",
                          "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct")
REVISION = os.environ.get("QUANT_REVISION",
                          "0ff6b5ec7c13b049b253a16a889aa269e6b79a94") or None

# 입력이 같은 선형층은 하나만 훅을 건다 (q/k/v 입력 동일, c_fc_0/c_fc_1 입력 동일).
HOOK_POINTS = {
    "attn_in": re.compile(r"\.h\.(\d+)\.attn\.attention\.q_proj$"),
    "attn_out": re.compile(r"\.h\.(\d+)\.attn\.attention\.out_proj$"),
    "mlp_in": re.compile(r"\.h\.(\d+)\.mlp\.c_fc_0$"),
    "mlp_out": re.compile(r"\.h\.(\d+)\.mlp\.c_proj$"),
}

KMMLU_SUBJECTS = ["Accounting", "Law", "Economics", "Korean-History",
                  "Computer-Science", "Biology", "Psychology", "Marketing"]
MMLU_SUBJECTS = ["high_school_macroeconomics", "professional_law", "econometrics",
                 "high_school_world_history", "college_computer_science",
                 "college_biology", "professional_psychology", "marketing"]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# 통계 계산 유틸 (순수 함수 — 단위 검증 대상)
# ---------------------------------------------------------------------------
def topk_set(vec: torch.Tensor, frac: float = 0.01) -> set:
    k = max(1, int(round(frac * vec.numel())))
    return set(torch.topk(vec, k).indices.tolist())


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if (a or b) else float("nan")


def spearman(x: torch.Tensor, y: torch.Tensor) -> float:
    """scipy 의존 없이 순위 상관을 계산한다(연속값이라 동률은 무시)."""
    rx = torch.argsort(torch.argsort(x.double())).double()
    ry = torch.argsort(torch.argsort(y.double())).double()
    rx -= rx.mean()
    ry -= ry.mean()
    den = (rx.norm() * ry.norm()).item()
    return (rx @ ry).item() / den if den > 0 else float("nan")


# ---------------------------------------------------------------------------
# 데이터
# ---------------------------------------------------------------------------
def calib_texts(lang: str, n: int, seed: int):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from run_quant_stage import load_calib_dataset
    return list(load_calib_dataset(lang, n, seed=seed)["text"])


def heldout_texts(lang: str, per_subject: int):
    """KMMLU/MMLU 문항을 평가 입력과 같은 형식으로 구성한다."""
    from datasets import load_dataset
    texts = []
    if lang == "ko":
        for s in KMMLU_SUBJECTS:
            try:
                ds = load_dataset("HAERAE-HUB/KMMLU", s, split="test")
            except Exception as e:
                log(f"  KMMLU {s} 로드 실패, 건너뜀: {type(e).__name__}")
                continue
            for ex in ds.select(range(min(per_subject, len(ds)))):
                texts.append(f"{ex['question']}\nA. {ex['A']}\nB. {ex['B']}\n"
                             f"C. {ex['C']}\nD. {ex['D']}\n정답:")
    else:
        for s in MMLU_SUBJECTS:
            try:
                ds = load_dataset("cais/mmlu", s, split="test")
            except Exception as e:
                log(f"  MMLU {s} 로드 실패, 건너뜀: {type(e).__name__}")
                continue
            for ex in ds.select(range(min(per_subject, len(ds)))):
                ch = ex["choices"]
                texts.append(f"{ex['question']}\nA. {ch[0]}\nB. {ch[1]}\n"
                             f"C. {ch[2]}\nD. {ch[3]}\nAnswer:")
    return texts


# ---------------------------------------------------------------------------
# 모델
# ---------------------------------------------------------------------------
def load_model(path_or_id, is_bf16):
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
    return m


def free(model):
    del model
    gc.collect()
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# M1·M2: 채널 통계 수집
# ---------------------------------------------------------------------------
@torch.no_grad()
def collect_channel_stats(model, tok, texts, max_len, max_layers=None):
    stats = {}      # (layer, point) -> [sum_abs, sum_sq, n]
    handles = []

    def make_hook(key):
        def hook(_mod, inputs):
            x = inputs[0].detach().float().reshape(-1, inputs[0].shape[-1])
            s = stats.get(key)
            if s is None:
                stats[key] = [x.abs().sum(0), (x * x).sum(0), x.shape[0]]
            else:
                s[0] += x.abs().sum(0)
                s[1] += (x * x).sum(0)
                s[2] += x.shape[0]
        return hook

    for name, mod in model.named_modules():
        for point, pat in HOOK_POINTS.items():
            m = pat.search(name)
            if m:
                layer = int(m.group(1))
                if max_layers is None or layer < max_layers:
                    handles.append(mod.register_forward_pre_hook(make_hook((layer, point))))

    if not handles:
        raise RuntimeError("훅을 걸 선형층을 찾지 못했다. 모듈 이름 규칙을 확인할 것.")

    # 배치 1, 패딩 없음: 패드 토큰이 통계를 오염시키지 않게 한다.
    for i, t in enumerate(texts):
        ids = tok(t, return_tensors="pt", truncation=True, max_length=max_len).input_ids.to("cuda:0")
        model(ids)
        if (i + 1) % 64 == 0:
            log(f"    {i + 1}/{len(texts)}")

    for h in handles:
        h.remove()
    return {k: (v[0] / v[2], v[1] / v[2]) for k, v in stats.items()}  # (mean_abs, mean_sq)


def compare_stats(a, b, frac):
    rows = {}
    for key in sorted(set(a) & set(b)):
        ja = jaccard(topk_set(a[key][0], frac), topk_set(b[key][0], frac))
        sp = spearman(a[key][1], b[key][1])
        rows[key] = (ja, sp)
    return rows


# ---------------------------------------------------------------------------
# M3: 레이어별 은닉 상태 오차
# ---------------------------------------------------------------------------
@torch.no_grad()
def hidden_states(model, ids):
    return model(ids, output_hidden_states=True).hidden_states


@torch.no_grad()
def layer_error(ref_model, q_model, tok, texts, max_len):
    """레이어별 Σ||h_q − h_ref||² / Σ||h_ref||² (텍스트·토큰 전체 누적)."""
    num, den = None, None
    for t in texts:
        ids = tok(t, return_tensors="pt", truncation=True, max_length=max_len).input_ids.to("cuda:0")
        hr = hidden_states(ref_model, ids)
        hq = hidden_states(q_model, ids)
        if num is None:
            num = torch.zeros(len(hr), dtype=torch.float64)
            den = torch.zeros(len(hr), dtype=torch.float64)
        for l, (a, b) in enumerate(zip(hr, hq)):
            a = a.float()
            b = b.float()
            num[l] += ((b - a) ** 2).sum().item()
            den[l] += (a ** 2).sum().item()
    return (num / den.clamp_min(1e-12)).tolist()


# ---------------------------------------------------------------------------
# 메인
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-calib", type=int, default=256)
    ap.add_argument("--calib-max-len", type=int, default=512)
    ap.add_argument("--heldout-per-subject", type=int, default=8)
    ap.add_argument("--heldout-max-len", type=int, default=256)
    ap.add_argument("--top-frac", type=float, default=0.01)
    ap.add_argument("--smoke", action="store_true", help="텍스트·레이어를 줄여 로딩만 점검")
    ap.add_argument("--skip-m12", action="store_true")
    ap.add_argument("--skip-m3", action="store_true")
    args = ap.parse_args()

    base, out = Path(args.base), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    n_calib = 4 if args.smoke else args.n_calib
    per_subj = 1 if args.smoke else args.heldout_per_subject
    max_layers = 2 if args.smoke else None

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION, trust_remote_code=True)

    log("BF16 모델 로딩")
    ref = load_model(MODEL_ID, is_bf16=True)

    summary = []

    # ---------------- M1·M2 ----------------
    if not args.skip_m12:
        sets = {
            "ko_s0": ("ko", 0), "ko_s1": ("ko", 1),
            "en_s0": ("en_instruct", 0), "en_s1": ("en_instruct", 1),
        }
        st = {}
        for name, (lang, seed) in sets.items():
            log(f"[M1·M2] 캘리브레이션 통계 수집: {name}")
            st[name] = collect_channel_stats(ref, tok, calib_texts(lang, n_calib, seed),
                                             args.calib_max_len, max_layers)

        pairs = {
            "ko_vs_ko(기준선)": ("ko_s0", "ko_s1"),
            "en_vs_en(기준선)": ("en_s0", "en_s1"),
            "ko_vs_en(검증)": ("ko_s0", "en_s0"),
        }
        comp = {p: compare_stats(st[a], st[b], args.top_frac) for p, (a, b) in pairs.items()}

        keys = sorted(comp["ko_vs_en(검증)"].keys())
        with open(out / "m1_m2_channels.csv", "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["layer", "point",
                        "jaccard_ko_ko", "jaccard_en_en", "jaccard_ko_en",
                        "spearman_ko_ko", "spearman_en_en", "spearman_ko_en"])
            for k in keys:
                w.writerow([k[0], k[1],
                            *[round(comp[p][k][0], 4) for p in pairs],
                            *[round(comp[p][k][1], 4) for p in pairs]])

        summary.append("=== M1 중요 채널 중첩도 (Jaccard, 상위 {:.0%}) ===".format(args.top_frac))
        summary.append("    기준선보다 '언어 간' 값이 뚜렷이 낮으면 언어별로 보호 대상 채널이 다르다는 근거")
        for point in HOOK_POINTS:
            vals = {p: [comp[p][k][0] for k in keys if k[1] == point] for p in pairs}
            summary.append("  {:9s} ".format(point) + "  ".join(
                f"{p}={sum(v) / len(v):.3f}" for p, v in vals.items() if v))

        summary.append("")
        summary.append("=== M2 헤시안 대각 순위 상관 (Spearman) ===")
        summary.append("    기준선보다 '언어 간' 값이 뚜렷이 낮으면 GPTQ의 보정 가중치가 언어에 따라 다르다는 근거")
        for point in HOOK_POINTS:
            vals = {p: [comp[p][k][1] for k in keys if k[1] == point] for p in pairs}
            summary.append("  {:9s} ".format(point) + "  ".join(
                f"{p}={sum(v) / len(v):.3f}" for p, v in vals.items() if v))
        summary.append("")
        del st
        gc.collect()

    # ---------------- M3 ----------------
    if not args.skip_m3:
        log("[M3] 보류 텍스트 로딩 (KMMLU·MMLU)")
        held = {"ko": heldout_texts("ko", per_subj), "en": heldout_texts("en", per_subj)}
        log(f"  한국어 {len(held['ko'])}개, 영어 {len(held['en'])}개")
        if not held["ko"] or not held["en"]:
            log("오류: 보류 텍스트를 확보하지 못했다. M3을 건너뛴다.")
        else:
            ckpts = {
                "gptq_ko": base / "gptq_4bit",
                "gptq_en_instruct": base / "gptq_4bit_calib_en_instruct",
                "awq_ko": base / "awq_4bit",
                "awq_en_instruct": base / "awq_4bit_calib_en_instruct",
            }
            err = {}
            for name, path in ckpts.items():
                if not path.is_dir():
                    log(f"  {name}: 체크포인트 없음, 건너뜀 → {path}")
                    continue
                log(f"[M3] {name} 로딩")
                try:
                    q = load_model(str(path), is_bf16=False)
                except Exception as e:
                    log(f"  {name} 로딩 실패: {type(e).__name__}: {e}")
                    continue
                for lang in ("ko", "en"):
                    log(f"  {name} × {lang}")
                    err[(name, lang)] = layer_error(ref, q, tok, held[lang], args.heldout_max_len)
                free(q)

            with open(out / "m3_layer_error.csv", "w", newline="", encoding="utf-8-sig") as f:
                w = csv.writer(f)
                w.writerow(["model", "eval_lang", "layer", "rel_error"])
                for (name, lang), vals in err.items():
                    for l, v in enumerate(vals):
                        w.writerow([name, lang, l, f"{v:.6e}"])

            summary.append("=== M3 언어별 레이어 출력 오차 (BF16 대비 상대 오차, 레이어 평균) ===")
            summary.append("    Δ_KO = 오차(en_instruct 캘리브, 한국어) − 오차(ko 캘리브, 한국어)  → 양수 예상")
            summary.append("    Δ_EN = 오차(ko 캘리브, 영어) − 오차(en_instruct 캘리브, 영어)    → 양수 예상")
            summary.append("    둘 다 양수면 정확도 교차와 같은 방향의 교차가 은닉 상태 수준에서 존재한다는 근거")
            for m in ("gptq", "awq"):
                k, e = f"{m}_ko", f"{m}_en_instruct"
                if all((x, l) in err for x in (k, e) for l in ("ko", "en")):
                    def mean(v):
                        v = v[1:]  # 임베딩 출력(0번)은 양자화 영향이 없으므로 제외
                        return sum(v) / len(v)
                    d_ko = mean(err[(e, "ko")]) - mean(err[(k, "ko")])
                    d_en = mean(err[(k, "en")]) - mean(err[(e, "en")])
                    n_layers = len(err[(k, "ko")]) - 1
                    pos_ko = sum(1 for l in range(1, n_layers + 1)
                                 if err[(e, "ko")][l] > err[(k, "ko")][l])
                    pos_en = sum(1 for l in range(1, n_layers + 1)
                                 if err[(k, "en")][l] > err[(e, "en")][l])
                    summary.append(f"  {m.upper():5s} Δ_KO={d_ko:+.3e} ({pos_ko}/{n_layers}층 양수)   "
                                   f"Δ_EN={d_en:+.3e} ({pos_en}/{n_layers}층 양수)")
                else:
                    summary.append(f"  {m.upper():5s} 체크포인트 누락으로 계산 불가")

    free(ref)
    text = "\n".join(summary)
    (out / "summary.txt").write_text(text + "\n", encoding="utf-8")
    print("\n" + text)
    log(f"결과 저장 → {out}")


if __name__ == "__main__":
    main()
