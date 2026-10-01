#!/usr/bin/env python3
"""
export_calib_manifests.py — 캘리브레이션 표본 기록 재구성·일괄 생성

표본 기록 기능은 실험 도중에 도입되었다. 그 이전에 실행된 양자화(seed 0~2, C4·혼합
조건, 기능 도입 전에 시작된 E2 작업)는 표본 기록이 없다. 표본 선택은 (언어, 크기, 시드)와
원본 데이터셋만으로 결정되고, 기록 기능 추가 전후의 선택 결과가 동일함을 검증했으므로
같은 규칙으로 다시 추출해 기록을 재구성한다.

재구성 기록은 meta.json의 origin 필드가 "reconstructed"로 표시되어 실행 중 기록
("runtime")과 구분된다. 원본 데이터셋의 리비전이 실행 당시와 같다는 전제가 필요하며,
이를 확인할 수 있도록 원본 캐시 파일 경로(리비전 해시 포함)를 함께 남긴다.

대상 조건을 찾는 방법
  1) 양자화 체크포인트 디렉터리 이름에서 추출 (기본)
  2) --jobs 로 지정한 작업 파일(sweep:METHOD:CALIB:SEED:N)에서 추출 — 아직 실행되지
     않은 계획 조건까지 미리 기록할 때

사용법:
  python export_calib_manifests.py --base quant_experiments/EXAONE-3.5-7.8B-Instruct
  python export_calib_manifests.py --base ... --jobs e23_jobs.txt
  python export_calib_manifests.py --base ... --no-text      # 본문 없이 식별자·해시만
"""

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_quant_stage import (NUM_CALIB_SAMPLES, calib_set_name,  # noqa: E402
                             load_calib_dataset, save_calib_manifest)

CKPT_RE = re.compile(r"^(awq|gptq|quarot)_4bit(?:_calib_(en_instruct|en|mixed))?(?:_n(\d+))?(?:_s(\d+))?$")


def from_checkpoints(base):
    conds = set()
    for d in base.iterdir():
        m = CKPT_RE.match(d.name)
        if m and d.is_dir():
            conds.add((m.group(2) or "ko", int(m.group(3) or NUM_CALIB_SAMPLES), int(m.group(4) or 0)))
    return conds


def from_jobs(paths):
    conds = set()
    for p in paths:
        for line in open(p, encoding="utf-8"):
            line = line.split("#")[0].strip()
            if line.startswith("sweep:"):
                _, _, calib, seed, n = line.split(":")
                conds.add((calib, int(n), int(seed)))
    return conds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--jobs", nargs="*", default=[])
    ap.add_argument("--no-text", action="store_true", help="본문 없이 식별자와 해시만 저장")
    ap.add_argument("--overwrite", action="store_true", help="이미 있는 기록도 다시 만든다")
    args = ap.parse_args()
    base = Path(args.base)
    out = base / "calib_manifests"

    conds = from_checkpoints(base) | from_jobs(args.jobs)
    print(f"대상 캘리브레이션 조건 {len(conds)}개")
    done = skipped = 0
    for lang, n, seed in sorted(conds):
        name = calib_set_name(lang, n, seed)
        meta = out / f"{name}.meta.json"
        if meta.exists() and not args.overwrite:
            origin = json.loads(meta.read_text(encoding="utf-8")).get("origin", "?")
            print(f"  건너뜀 {name} (이미 있음, origin={origin})")
            skipped += 1
            continue
        manifest, provenance = [], {}
        load_calib_dataset(lang, n, seed=seed, manifest=manifest, provenance=provenance)
        save_calib_manifest(out, lang, n, seed, manifest, provenance,
                            include_text=not args.no_text, origin="reconstructed")
        done += 1
    print(f"\n생성 {done}개, 건너뜀 {skipped}개 → {out}")
    print("주의: 원본 데이터셋 리비전이 실행 당시와 같아야 재구성이 정확하다.")
    print("      meta.json의 sources.*.cache_file 경로에 리비전 해시가 포함되어 있다.")


if __name__ == "__main__":
    main()
