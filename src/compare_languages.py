#!/usr/bin/env python3
"""
compare_languages.py
- 한국어(eval_*)와 영어(eval_en_*) 평가 결과를 나란히 놓고,
  BF16 대비 열화 폭이 언어에 따라 다른지(언어별 열화 비대칭) 분석한다.
- 난이도를 통제하기 위해 대응 관계가 있는 태스크 쌍을 중심으로 비교한다:
      KMMLU            <-> MMLU            (다지선다 지식)
      KoBEST HellaSwag <-> HellaSwag       (상식적 문장 완성)
      KoBEST BoolQ     <-> BoolQ           (참/거짓 추론)
  ARC-Challenge는 대응하는 한국어 태스크가 없어 영어 단독 참고치로만 둔다.

사용법:
    python compare_languages.py --output-dir quant_experiments/EXAONE-3.5-7.8B-Instruct
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

METHOD_DIR_MAP = {
    "bf16": "BF16",
    "awq": "AWQ-4bit[calib:ko]",
    "gptq": "GPTQ-4bit[calib:ko]",
    "awq_calib_en": "AWQ-4bit[calib:en]",
    "gptq_calib_en": "GPTQ-4bit[calib:en]",
    "awq_calib_en_instruct": "AWQ-4bit[calib:en_instruct]",
    "gptq_calib_en_instruct": "GPTQ-4bit[calib:en_instruct]",
    "awq_calib_mixed": "AWQ-4bit[calib:mixed]",
    "gptq_calib_mixed": "GPTQ-4bit[calib:mixed]",
    "quarot_rtn_backup": "QuaRot-4bit(RTN)",
    "quarot": "QuaRot-4bit(GPTQ)[calib:ko]",
    "quarot_calib_en": "QuaRot-4bit(GPTQ)[calib:en]",
}

# (쌍 이름, 한국어 태스크, 영어 태스크)
TASK_PAIRS = [
    ("MMLU계열", "kmmlu", "mmlu"),
    ("HellaSwag", "kobest_hellaswag", "hellaswag"),
    ("BoolQ", "kobest_boolq", "boolq"),
]

# 대응 쌍이 없어 참고용으로만 수집하는 태스크
KO_ONLY = ["kobest_copa", "kobest_wic", "kobest_sentineg", "haerae"]
EN_ONLY = ["arc_challenge"]

METRIC_PRIORITY = ["acc_norm,none", "acc,none", "f1,none", "exact_match,none"]


def find_results_json(eval_dir: Path):
    if not eval_dir.exists():
        return None
    candidates = sorted(eval_dir.rglob("results*.json"))
    return candidates[-1] if candidates else None


def extract_scores(results_json: Path, tasks) -> dict:
    with open(results_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    results = data.get("results", {})
    scores = {}
    for task in tasks:
        if task not in results:
            scores[task] = None
            continue
        tm = results[task]
        value = None
        for metric in METRIC_PRIORITY:
            if metric in tm:
                value = tm[metric]
                break
        if value is None:
            for k, v in tm.items():
                if k.endswith(",none") and isinstance(v, (int, float)):
                    value = v
                    break
        scores[task] = value
    return scores


def collect(base: Path, prefix: str, tasks) -> dict:
    """prefix가 'eval_'이면 한국어, 'eval_en_'이면 영어."""
    out = {}
    for key, label in METHOD_DIR_MAP.items():
        results_json = find_results_json(base / f"{prefix}{key}")
        if results_json is None:
            print(f"[건너뜀] {label} ({prefix}): 결과 파일 없음")
            continue
        out[label] = extract_scores(results_json, tasks)
        print(f"[확인] {label} ({prefix}): {results_json.name}")
    return out


def degradation(df: pd.DataFrame) -> pd.DataFrame:
    """BF16 대비 상대 저하율(%). 양수 = BF16보다 낮음."""
    if "BF16" not in df.index:
        return None
    bf16 = df.loc["BF16"]
    deg = (bf16 - df) / bf16 * 100
    deg.loc["BF16"] = 0.0
    return deg


def main():
    parser = argparse.ArgumentParser(description="한국어-영어 열화 비대칭 분석")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    base = Path(args.output_dir)
    if not base.exists():
        print(f"오류: {base} 디렉터리가 없다.", file=sys.stderr)
        sys.exit(1)

    ko_tasks = [ko for _, ko, _ in TASK_PAIRS] + KO_ONLY
    en_tasks = [en for _, _, en in TASK_PAIRS] + EN_ONLY

    print("--- 한국어 결과 수집 ---")
    ko = collect(base, "eval_", ko_tasks)
    print("\n--- 영어 결과 수집 ---")
    en = collect(base, "eval_en_", en_tasks)

    if not ko or not en:
        print("\n한쪽 언어의 결과가 없어 비교할 수 없다.")
        sys.exit(0)

    ko_df = pd.DataFrame(ko).T[ko_tasks]
    en_df = pd.DataFrame(en).T[en_tasks]

    ko_deg = degradation(ko_df)
    en_deg = degradation(en_df)

    print("\n" + "=" * 70)
    print("=== 한국어 원점수 ===")
    print(ko_df.round(4).to_string())
    print("\n=== 영어 원점수 ===")
    print(en_df.round(4).to_string())

    if ko_deg is None or en_deg is None:
        print("\nBF16 결과가 없어 저하율을 계산하지 못했다.")
        sys.exit(0)

    print("\n=== 한국어 저하율 (%) ===")
    print(ko_deg.round(2).to_string())
    print("\n=== 영어 저하율 (%) ===")
    print(en_deg.round(2).to_string())

    # 핵심 표: 대응 쌍별로 한국어 저하율 - 영어 저하율
    print("\n" + "=" * 70)
    print("=== 언어별 열화 비대칭 (한국어 저하율 − 영어 저하율, %p) ===")
    print("    단위: 퍼센트포인트(%p). 두 저하율(%)의 차이이므로 %가 아니라 %p로 표기한다.")
    print("    양수 = 한국어 저하율이 영어보다 크다 (이 조건에서 한국어가 상대적으로 불리)")
    print("    주의: 절대 성능 차이가 아니라 각 언어의 BF16 대비 상대 저하율의 차이다.")
    print("          또한 단일 조건의 관측치이므로, 표준오차를 고려하지 않은 작은 값")
    print("          (대략 ±1%p 내외)은 유의한 차이로 해석하지 않는 것이 안전하다.")
    rows = {}
    for pair_name, ko_task, en_task in TASK_PAIRS:
        col = {}
        for method in ko_deg.index:
            if method not in en_deg.index:
                continue
            k = ko_deg.loc[method, ko_task]
            e = en_deg.loc[method, en_task]
            col[method] = None if pd.isna(k) or pd.isna(e) else k - e
        rows[pair_name] = col
    asym = pd.DataFrame(rows)
    print(asym.round(2).to_string())

    # 저장
    ko_df.round(4).to_csv(base / "lang_ko_scores.csv", encoding="utf-8-sig")
    en_df.round(4).to_csv(base / "lang_en_scores.csv", encoding="utf-8-sig")
    ko_deg.round(2).to_csv(base / "lang_ko_degradation.csv", encoding="utf-8-sig")
    en_deg.round(2).to_csv(base / "lang_en_degradation.csv", encoding="utf-8-sig")
    asym.round(2).to_csv(base / "lang_asymmetry.csv", encoding="utf-8-sig")
    print(f"\n저장 완료 → {base}/lang_*.csv (5개 파일)")


if __name__ == "__main__":
    main()
