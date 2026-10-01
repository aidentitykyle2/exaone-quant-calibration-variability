#!/usr/bin/env python3
"""
aggregate_results.py
- quant_experiments/<모델명>/eval_{bf16,awq,gptq,quarot,spinquant}/ 안의
  lm-evaluation-harness 결과 JSON을 찾아 하나의 표로 취합한다.
- BF16 대비 상대 저하율(%)을 함께 계산한다.
- CSV(논문 표로 바로 옮기기 좋은 형태)와 사람이 읽기 좋은 콘솔 표를 둘 다 낸다.

사용법:
    python aggregate_results.py --output-dir quant_experiments/EXAONE-3.5-7.8B-Instruct
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

# EVAL_TASKS와 동일한 상위 태스크만 취합한다 — kmmlu 내부 45개+ 세부 과목까지
# 전부 펼치면 표가 논문에 쓰기엔 너무 커지므로, 상위 집계값만 우선 다룬다.
# 세부 과목별 분석이 필요해지면 이 리스트에 kmmlu_<과목명>을 추가하면 된다.
TOP_LEVEL_TASKS = [
    "kmmlu",
    "kobest_boolq",
    "kobest_copa",
    "kobest_wic",
    "kobest_hellaswag",
    "kobest_sentineg",
    "haerae",
]

# 디렉터리 키(eval_<키>) → 표에 표시할 기법 이름.
# QuaRot은 회전 이후 양자화 방식에 따라 두 조건을 모두 비교한다:
#   - QuaRot-4bit(RTN): 회전 + QuantizationModifier(data-free round-to-nearest,
#                       캘리브레이션 데이터 0개). 초기 구현으로, 백업 디렉터리
#                       eval_quarot_rtn_backup 에 보존되어 있다.
#   - QuaRot-4bit(GPTQ): 회전 + GPTQModifier(256샘플 Hessian 기반 오차 보정).
#                        QuaRot 원 논문의 구성이며 AWQ/GPTQ와 공정 비교 가능.
# 이 둘과 GPTQ 단독을 함께 놓으면 "회전의 기여"와 "오차 보정의 기여"를
# 분리해 볼 수 있는 ablation 표가 된다.
METHOD_DIR_MAP = {
    "bf16": "BF16",
    "awq": "AWQ-4bit",
    "gptq": "GPTQ-4bit",
    "quarot_rtn_backup": "QuaRot-4bit(RTN)",
    "quarot": "QuaRot-4bit(GPTQ)",
    "spinquant": "SpinQuant-4bit",
}

# 태스크 유형별로 lm-evaluation-harness가 주로 쓰는 기본 지표 우선순위.
# 태스크마다 존재하는 지표가 다를 수 있어 순서대로 탐색해 처음 찾은 값을 쓴다.
METRIC_PRIORITY = ["acc_norm,none", "acc,none", "f1,none", "exact_match,none"]


def find_results_json(eval_dir: Path) -> Path | None:
    """output_path 아래 어딘가에 있는 results*.json을 찾는다.
    lm-evaluation-harness 버전에 따라 파일명(타임스탬프 포함 여부),
    중첩 디렉터리 구조(모델명 하위 폴더 생성 여부)가 달라질 수 있어
    재귀적으로 탐색한다."""
    if not eval_dir.exists():
        return None
    candidates = sorted(eval_dir.rglob("results*.json"))
    if not candidates:
        return None
    # 여러 개면 가장 최근 것(파일명에 타임스탬프가 있는 경우 사전순 마지막)
    return candidates[-1]


def extract_scores(results_json: Path) -> dict:
    with open(results_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    scores = {}
    results = data.get("results", {})
    for task in TOP_LEVEL_TASKS:
        if task not in results:
            scores[task] = None
            continue
        task_metrics = results[task]
        value = None
        used_metric = None
        for metric in METRIC_PRIORITY:
            if metric in task_metrics:
                value = task_metrics[metric]
                used_metric = metric
                break
        if value is None:
            # 위 우선순위에 없는 지표뿐이면, ",none"으로 끝나는 첫 숫자형 지표를 그냥 채택
            for k, v in task_metrics.items():
                if k.endswith(",none") and isinstance(v, (int, float)):
                    value, used_metric = v, k
                    break
        scores[task] = value
        if used_metric:
            scores.setdefault("_metric_used", {})[task] = used_metric
    return scores


def main():
    parser = argparse.ArgumentParser(description="양자화 기법별 평가 결과 취합")
    parser.add_argument("--output-dir", required=True,
                         help="quant_experiments/<모델명> 디렉터리")
    args = parser.parse_args()

    base = Path(args.output_dir)
    if not base.exists():
        print(f"오류: {base} 디렉터리가 없다.", file=sys.stderr)
        sys.exit(1)

    all_scores = {}
    metric_notes = {}
    for method_key, method_label in METHOD_DIR_MAP.items():
        eval_dir = base / f"eval_{method_key}"
        results_json = find_results_json(eval_dir)
        if results_json is None:
            print(f"[건너뜀] {method_label}: {eval_dir} 에서 결과 파일을 찾지 못했다 "
                  f"(아직 평가 전이거나 진행 중일 수 있다)")
            continue
        scores = extract_scores(results_json)
        metric_notes[method_label] = scores.pop("_metric_used", {})
        all_scores[method_label] = scores
        print(f"[확인] {method_label}: {results_json}")

    if not all_scores:
        print("취합할 결과가 하나도 없다. 평가가 끝난 뒤 다시 실행할 것.")
        sys.exit(0)

    df = pd.DataFrame(all_scores).T  # 행: 기법, 열: 태스크
    df = df[[t for t in TOP_LEVEL_TASKS if t in df.columns]]

    print("\n=== 원점수 (accuracy) ===")
    print(df.round(4).to_string())

    if "BF16" in df.index:
        bf16 = df.loc["BF16"]
        degradation = ((bf16 - df) / bf16 * 100)
        degradation.loc["BF16"] = 0.0  # 기준선 자기 자신은 0%
        print("\n=== BF16 대비 저하율 (%, 양수=BF16보다 낮음) ===")
        print(degradation.round(2).to_string())
    else:
        degradation = None
        print("\nBF16 결과가 아직 없어 저하율은 계산하지 못했다.")

    out_csv = base / "results_summary.csv"
    df.round(4).to_csv(out_csv, encoding="utf-8-sig")
    print(f"\n원점수 저장 → {out_csv}")

    if degradation is not None:
        out_csv_deg = base / "results_degradation.csv"
        degradation.round(2).to_csv(out_csv_deg, encoding="utf-8-sig")
        print(f"저하율 저장 → {out_csv_deg}")

    # 어떤 지표(acc vs acc_norm 등)를 태스크별로 썼는지도 남겨, 논문 표 각주에 활용
    metric_notes_path = base / "results_metrics_used.json"
    with open(metric_notes_path, "w", encoding="utf-8") as f:
        json.dump(metric_notes, f, indent=2, ensure_ascii=False)
    print(f"사용된 지표 기록 → {metric_notes_path}")


if __name__ == "__main__":
    main()
