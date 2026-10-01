#!/usr/bin/env python3
"""
export_item_scores.py — 문항별 채점 결과를 재현 가능한 소형 배열로 내보낸다

왜 필요한가
-----------
논문의 두 단계 부트스트랩(4.1절)과 라벨 재배정 순열 검정(4.2절)은 **문항별
점수**가 있어야 재현된다. 그 원천은 lm-eval의 `--log_samples` 기록이지만,
KMMLU 35,030문항 × 수십 회 실행이면 수 GB라 저장소에 올릴 수 없다.

이 스크립트는 각 실행·과제에 대해 다음만 남긴다.

  first  : 동률이면 첫 선택 규칙의 정답 여부 (0/1)
  hit    : 정답이 동률 집합에 포함되는지 (0/1)
  ntied  : 동률 선택지 개수 (1이면 동률 없음)

동률 분할 점수는 `hit / ntied`로 **정확히** 복원된다. 따라서 두 채점 규칙
(4.7절의 1.87배 대비)을 모두 재현할 수 있으면서, 저장 크기는 원본의 1/1000
수준이다. 길이 정규화 과제(HellaSwag 계열, ARC-Challenge)는 원시 로그우도로
재채점할 수 없으므로 기록된 acc_norm을 float32로 그대로 담는다.

채점 논리는 `rescore_all.py`를 import해 공유한다. 논문 표의 값과 다른 경로로
계산하면 두 산출물이 어긋날 수 있으므로, 함수를 복제하지 않는다.

문항 정렬
---------
과제별로 `subtask:doc_id`를 사전순으로 정렬해 표준 순서를 정하고 그 순서의
sha256을 함께 저장한다. 실행마다 순서가 같은지 확인할 수 있고, 다르면
대응 부트스트랩이 무효이므로 즉시 드러난다.

사용법
------
  python export_item_scores.py --base quant_experiments/EXAONE-3.5-7.8B-Instruct \\
      --out data/per_item_scores
  # -> <out>/<모델명>_<lang>.npz  와  <out>/<모델명>_<lang>.index.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rescore_all import (  # noqa: E402  채점 논리를 논문 표와 공유한다
    ACC_NORM_TASKS,
    SAMPLE_RE,
    TASKS,
    match,
    target_index,
)


def collect_run(run_dir: Path, lang: str):
    """한 실행 디렉터리에서 과제별 {문항키: (first, hit, ntied) 또는 acc_norm}을 모은다."""
    latest = {}
    for p in run_dir.rglob("samples_*.jsonl"):
        m = SAMPLE_RE.search(p.name)
        if m and (m.group(1) not in latest or m.group(2) > latest[m.group(1)][0]):
            latest[m.group(1)] = (m.group(2), p)

    per_task = defaultdict(dict)
    dropped = defaultdict(int)
    for subtask, (_, p) in latest.items():
        name = next((n for n, how, v in TASKS[lang] if match(subtask, how, v)), None)
        if name is None:
            continue
        with open(p, encoding="utf-8") as f:
            for pos, line in enumerate(f):
                r = json.loads(line)
                fr = r.get("filtered_resps")
                if not fr:
                    continue
                key = f"{subtask}:{r.get('doc_id', pos)}"
                if name in ACC_NORM_TASKS:
                    per_task[name][key] = float(r.get("acc_norm", r.get("acc", 0.0)))
                    continue
                ll = [float(x[0]) if isinstance(x, (list, tuple)) else float(x) for x in fr]
                mx = max(ll)
                tied = [i for i, v in enumerate(ll) if v == mx]
                t = target_index(r)
                if t is None:
                    dropped[name] += 1
                    continue
                per_task[name][key] = (
                    int(ll.index(mx) == t), int(t in tied), int(len(tied)))
    return per_task, dict(dropped)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, required=True,
                    help="모델 실험 디렉터리 (seed_eval/ 를 포함)")
    ap.add_argument("--out", type=Path, required=True, help="npz 저장 위치")
    ap.add_argument("--langs", default="ko,en")
    a = ap.parse_args()

    model = a.base.name
    a.out.mkdir(parents=True, exist_ok=True)
    rc = 0

    for lang in a.langs.split(","):
        root = a.base / "seed_eval" / lang
        if not root.is_dir():
            print(f"[{lang}] {root} 가 없다 — 건너뛴다")
            continue
        runs = sorted(d for d in root.iterdir() if d.is_dir())
        if not runs:
            print(f"[{lang}] 실행 디렉터리가 없다")
            continue

        gathered, dropped_all = {}, {}
        for d in runs:
            per_task, dropped = collect_run(d, lang)
            if per_task:
                gathered[d.name] = per_task
            if dropped:
                dropped_all[d.name] = dropped

        # 과제별 표준 문항 순서: 모든 실행에 공통으로 존재하는 문항만 쓴다.
        arrays, index = {}, {"model": model, "lang": lang, "runs": sorted(gathered), "tasks": {}}
        for name, _, _ in TASKS[lang]:
            present = [r for r in gathered if name in gathered[r]]
            if not present:
                continue
            common = set(gathered[present[0]][name])
            for r in present[1:]:
                common &= set(gathered[r][name])
            order = sorted(common)
            if not order:
                continue
            sizes = {r: len(gathered[r][name]) for r in present}
            if len(set(sizes.values())) > 1 or len(order) != min(sizes.values()):
                # 실행 간 문항 집합이 다르면 대응 부트스트랩의 전제가 깨진다.
                print(f"[{lang}/{name}] 경고: 실행 간 문항 수가 다르다 "
                      f"(공통 {len(order)}, 실행별 {sorted(set(sizes.values()))})")
                rc = 2
            h = hashlib.sha256("\n".join(order).encode()).hexdigest()
            arrays[f"order__{name}"] = np.array(order, dtype=object)
            index["tasks"][name] = {"n_items": len(order), "order_sha256": h,
                                    "runs": present,
                                    "scoring": "acc_norm" if name in ACC_NORM_TASKS
                                    else "loglikelihood(first, hit/ntied)"}
            for r in present:
                d = gathered[r][name]
                if name in ACC_NORM_TASKS:
                    arrays[f"{r}__{name}__acc_norm"] = np.array(
                        [d[k] for k in order], dtype=np.float32)
                else:
                    arrays[f"{r}__{name}__first"] = np.array(
                        [d[k][0] for k in order], dtype=np.int8)
                    arrays[f"{r}__{name}__hit"] = np.array(
                        [d[k][1] for k in order], dtype=np.int8)
                    arrays[f"{r}__{name}__ntied"] = np.array(
                        [d[k][2] for k in order], dtype=np.int8)

        if not arrays:
            print(f"[{lang}] 내보낼 것이 없다")
            continue
        if dropped_all:
            index["dropped_items"] = dropped_all

        npz = a.out / f"{model}_{lang}.npz"
        np.savez_compressed(npz, **arrays)
        (a.out / f"{model}_{lang}.index.json").write_text(
            json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
        mb = npz.stat().st_size / 2**20
        print(f"[{lang}] 실행 {len(gathered)}개 · 과제 {len(index['tasks'])}개 → "
              f"{npz} ({mb:.2f} MB)")
        for t, meta in index["tasks"].items():
            print(f"       {t}: {meta['n_items']}문항, 순서 sha256 {meta['order_sha256'][:12]}")

    print("\n동률 분할 점수는 hit / ntied 로 정확히 복원된다.")
    print("논문 표와의 대조는 rescore_all.py 의 출력과 비교해 확인할 것.")
    return rc


if __name__ == "__main__":
    sys.exit(main())
