#!/usr/bin/env python3
"""
item_stability.py — 문항 수준 손상이 시드에 따라 이동하는지 측정한다

왜 필요한가.

final_analysis.py의 A절은 "상위 10% 문항이 합계에서 차지하는 비중이 높을수록
시드 간 변동이 커진다"고 해석한다. 그런데 32B 실측은 그 반대다.

  MMLU  꼬리 집중도 70.3 / 71.2%   그런데 SD(Δ_EN) = 0.38  (안정)
  KMMLU 꼬리 집중도 45.9 / 44.6%   그런데 SD(Δ_KO) = 1.49  (불안정)

집중도만으로는 설명되지 않는다. 집중도가 변동을 만들려면 **손상되는 문항이
시드마다 달라져야** 한다. 같은 소수 문항이 항상 손상되면 집중도가 높아도
평균은 안정적이다.

따라서 진짜 물어야 할 것은 집중도가 아니라 손상 문항 집합의 이동이다.
이 스크립트는 세 가지를 센다.

  S1 순위 상관      시드 쌍 사이의 문항별 Choice-KL 스피어만 상관.
                    높으면 같은 문항이 늘 손상된다.
  S2 상위 집합 겹침 상위 10% 손상 문항 집합의 자카드 겹침(시드 쌍 평균).
  S3 동률-KL 상관   시드별 동률률과 시드별 평균 Choice-KL의 상관.

S3의 목적은 final_analysis.py C절의 교란을 제거하는 것이다. C절은 동률률과
tie_split 저하율의 상관을 보는데, tie_split은 동률 문항에 1/k를 주므로
동률이 늘면 정확도가 기계적으로 우연 수준(0.25)에 끌린다. BF16 정확도가
약 0.50이므로 동률 증가는 정의상 저하율을 키운다. 즉 C절의 상관에는
인과가 아닌 정의가 섞여 있다. Choice-KL은 동률 처리를 전혀 쓰지 않는
분포 거리이므로, 동률률이 Choice-KL과도 상관되면 그때는 '모델 확신 구조'
설명이 채점 규칙과 무관하게 성립한다.

예상되는 결과와 그 해석
  MMLU에서 S1·S2가 높고 KMMLU에서 낮으면
    -> 영어 손상은 고정된 문항 집합에 몰리고, 한국어 손상은 캘리브레이션
       표본에 따라 옮겨 다닌다. Δ_EN이 거의 상수인 이유가 직접 설명된다.
  두 벤치마크의 S1·S2가 비슷하면
    -> 문항 집합 이동으로도 설명되지 않는다. 그때는 A절을 기각하고,
       변동의 원인을 문항 구조가 아닌 다른 곳에서 찾아야 한다.
  어느 쪽이든 논문 6장에 쓸 결과가 나온다.

사용법
  python item_stability.py --base quant_experiments/EXAONE-3.5-32B-Instruct
  python item_stability.py --base ... --tasks kmmlu mmlu --top-frac 0.10
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# 평가 언어와 벤치마크의 대응 (seed_job.sh의 저장 구조)
TASK_LANG = {"kmmlu": "ko", "mmlu": "en"}


# ------------------------------------------------------------------
# samples jsonl 읽기
# ------------------------------------------------------------------
def choice_logprobs(rec: dict) -> Optional[List[float]]:
    """filtered_resps에서 선택지별 로그우도를 뽑는다. 형식 변형을 모두 받는다."""
    fr = rec.get("filtered_resps")
    if fr is None:
        return None
    out = []
    for x in fr:
        if isinstance(x, (list, tuple)):
            if not x:
                return None
            out.append(float(x[0]))
        elif isinstance(x, dict):
            for k in ("logprob", "loglikelihood", "score"):
                if k in x:
                    out.append(float(x[k]))
                    break
            else:
                return None
        else:
            out.append(float(x))
    return out if len(out) >= 2 else None


def item_key(rec: dict, task: str) -> str:
    """문항 식별자. doc_id는 서브태스크 안에서만 유일하므로 태스크명을 붙인다."""
    did = rec.get("doc_id")
    if did is None:
        did = rec.get("doc", {}).get("id") if isinstance(rec.get("doc"), dict) else None
    return f"{task}#{did}"


def load_condition(dirpath: Path) -> Dict[str, np.ndarray]:
    """
    한 조건의 seed_eval 디렉터리에서 {문항키: 선택지 로그우도 배열}을 만든다.
    서브태스크별 samples 파일이 여러 개이므로 전부 합친다.
    """
    items: Dict[str, np.ndarray] = {}
    files = sorted(dirpath.rglob("samples_*.jsonl"))
    if not files:
        return items
    bad = 0
    for f in files:
        # samples_<subtask>_<timestamp>.jsonl 에서 서브태스크명을 꺼낸다
        stem = f.stem
        sub = stem[len("samples_"):].rsplit("_", 1)[0] if stem.startswith("samples_") else stem
        with f.open(encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    bad += 1
                    continue
                ll = choice_logprobs(rec)
                if ll is None:
                    bad += 1
                    continue
                items[item_key(rec, sub)] = np.asarray(ll, dtype=np.float64)
    if bad:
        print(f"    [주의] {dirpath.name}: 해석하지 못한 줄 {bad}개")
    return items


def softmax(x: np.ndarray) -> np.ndarray:
    z = x - x.max()
    e = np.exp(z)
    return e / e.sum()


def tie_frac(items: Dict[str, np.ndarray]) -> float:
    n = t = 0
    for ll in items.values():
        n += 1
        mx = ll.max()
        if int((ll == mx).sum()) > 1:
            t += 1
    return t / n if n else float("nan")


# ------------------------------------------------------------------
# 지표
# ------------------------------------------------------------------
def per_item_kl(ref: Dict[str, np.ndarray], cmp_: Dict[str, np.ndarray],
                keys: List[str]) -> np.ndarray:
    """문항별 Choice-KL: KL(P_quant || P_BF16), 선택지 4개로 제한된 분포."""
    out = np.empty(len(keys))
    for i, k in enumerate(keys):
        p, q = softmax(cmp_[k]), softmax(ref[k])
        m = p > 0
        out[i] = float((p[m] * (np.log(p[m]) - np.log(np.clip(q[m], 1e-300, None)))).sum())
    return out


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra, rb = rankdata(a), rankdata(b)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    d = math.sqrt(float((ra ** 2).sum()) * float((rb ** 2).sum()))
    return float((ra * rb).sum() / d) if d > 0 else float("nan")


def rankdata(a: np.ndarray) -> np.ndarray:
    """동순위 평균 랭크 (scipy 없이)."""
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=np.float64)
    sorted_a = a[order]
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and sorted_a[j + 1] == sorted_a[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    d = math.sqrt(float((a ** 2).sum()) * float((b ** 2).sum()))
    return float((a * b).sum() / d) if d > 0 else float("nan")


def jaccard_top(vs: List[np.ndarray], frac: float) -> float:
    """상위 frac 문항 집합의 시드 쌍 평균 자카드 겹침."""
    k = max(1, int(round(len(vs[0]) * frac)))
    sets = [set(np.argsort(-v)[:k].tolist()) for v in vs]
    vals = [len(a & b) / len(a | b) for a, b in combinations(sets, 2)]
    return float(np.mean(vals)) if vals else float("nan")


def r_ci(r: float, n: int) -> Tuple[float, float]:
    """Fisher z 기반 상관계수 95% 구간. n이 작을 때 구간이 얼마나 넓은지 보이기 위함."""
    if n < 4 or not np.isfinite(r) or abs(r) >= 1:
        return (float("nan"), float("nan"))
    z = 0.5 * math.log((1 + r) / (1 - r))
    se = 1.0 / math.sqrt(n - 3)
    lo, hi = z - 1.959964 * se, z + 1.959964 * se
    return (math.tanh(lo), math.tanh(hi))


# ------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--tasks", nargs="+", default=["kmmlu", "mmlu"])
    ap.add_argument("--calibs", nargs="+", default=["ko", "en_instruct"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5, 6])
    ap.add_argument("--top-frac", type=float, default=0.10)
    ap.add_argument("--out", default=None, help="결과 JSON 경로")
    a = ap.parse_args()

    base = Path(a.base)
    report = {}

    print("=" * 74)
    print("문항 수준 손상의 시드 간 안정성")
    print("=" * 74)

    for task in a.tasks:
        evlang = TASK_LANG.get(task)
        if evlang is None:
            print(f"[{task}] 평가 언어를 모른다. TASK_LANG에 추가할 것.")
            continue
        root = base / "seed_eval" / evlang
        print(f"\n### {task.upper()}  ({root})")

        ref = load_condition(root / "bf16")
        if not ref:
            print("  BF16 samples를 찾지 못했다. 건너뛴다.")
            continue

        for calib in a.calibs:
            conds, tie_rates, names = [], [], []
            for s in a.seeds:
                d = root / f"gptq_{calib}_s{s}"
                it = load_condition(d)
                if not it:
                    print(f"  [없음] {d}")
                    continue
                conds.append(it)
                tie_rates.append(tie_frac(it))
                names.append(f"s{s}")
            if len(conds) < 2:
                print(f"  {calib}: 조건이 2개 미만이라 비교할 수 없다.")
                continue

            # 모든 조건과 BF16에 공통으로 존재하는 문항만 쓴다
            keys = set(ref.keys())
            for c in conds:
                keys &= set(c.keys())
            keys = sorted(keys)
            if not keys:
                print(f"  {calib}: 공통 문항이 없다.")
                continue

            vecs = [per_item_kl(ref, c, keys) for c in conds]
            means = np.array([v.mean() for v in vecs])

            # S1 순위 상관
            rs = [spearman(x, y) for x, y in combinations(vecs, 2)]
            s1 = float(np.mean(rs))
            # S2 상위 집합 겹침
            s2 = jaccard_top(vecs, a.top_frac)
            # 무작위 기대 겹침 (참조선): frac/(2-frac)
            base_j = a.top_frac / (2 - a.top_frac)
            # S3 동률률-평균KL 상관
            tr = np.array(tie_rates)
            s3 = pearson(tr, means)
            lo3, hi3 = r_ci(s3, len(tr))

            print(f"  {calib:12s} 문항 {len(keys)} · 조건 {len(conds)} ({','.join(names)})")
            print(f"    S1 시드 쌍 순위상관   {s1:.3f}   (범위 {min(rs):.3f}~{max(rs):.3f})")
            print(f"    S2 상위{a.top_frac*100:.0f}% 집합 겹침  {s2:.3f}   "
                  f"(무작위 기대 {base_j:.3f})")
            print(f"    S3 동률률-평균KL 상관 {s3:+.3f}   "
                  f"95% [{lo3:+.3f}, {hi3:+.3f}]  (점 {len(tr)}개)")
            print(f"       동률률 {tr.min()*100:.1f}~{tr.max()*100:.1f}% · "
                  f"평균 Choice-KL {means.min():.5f}~{means.max():.5f}")

            report[f"{task}/{calib}"] = {
                "n_items": len(keys), "seeds": names,
                "pair_spearman_mean": s1, "pair_spearman_range": [min(rs), max(rs)],
                "top_jaccard": s2, "top_jaccard_random": base_j,
                "tie_kl_pearson": s3, "tie_kl_pearson_ci": [lo3, hi3],
                "tie_rates": tie_rates, "mean_kl": means.tolist(),
            }

    print("\n" + "=" * 74)
    print("판정 지침")
    print("  S1·S2가 벤치마크 간에 크게 다르면 -> 손상 문항 집합의 이동이")
    print("     Δ_KO와 Δ_EN의 안정성 차이를 설명한다. 6장에 그대로 쓸 수 있다.")
    print("  S1·S2가 비슷하면 -> final_analysis.py A절의 꼬리 집중도 해석을")
    print("     기각하고, 변동의 원인을 문항 구조 밖에서 찾아야 한다.")
    print("  S3의 구간이 0을 포함하면 -> C절의 동률-저하율 상관은 채점 규칙에")
    print("     따른 정의적 관계일 가능성이 남는다. 그렇게 서술할 것.")

    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(report, ensure_ascii=False, indent=2),
                               encoding="utf-8")
        print(f"\n저장 → {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
