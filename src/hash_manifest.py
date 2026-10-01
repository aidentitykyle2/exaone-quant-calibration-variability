#!/usr/bin/env python3
"""
hash_manifest.py — 저장소 전체 파일의 sha256 목록을 만든다

재현성 주장을 하는 논문의 저장소가 정작 "지금 올려둔 것이 논문에 쓴 그것인가"를
증명하지 못하면 곤란하다. 이 스크립트는 커밋 대상 파일 전부의 해시를 한 파일에
모아, 나중에 저장소가 바뀌었는지 확인할 수 있게 한다.

.gitignore 규칙은 해석하지 않는다. 대신 명시적 제외 목록을 쓴다 — git이 무엇을
무시하는지와 무관하게, 이 목록이 "배포물에 무엇이 들어갔는가"의 기록이 된다.

사용법:
  python src/hash_manifest.py --root . --out MANIFEST.sha256
  python src/hash_manifest.py --root . --check MANIFEST.sha256
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

EXCLUDE_DIRS = {".git", "__pycache__", "venv", "venv-quant", "quant_experiments",
                "seed_eval", "queue_claims", ".ipynb_checkpoints", "node_modules"}
EXCLUDE_SUFFIX = {".pyc", ".bak", ".safetensors", ".bin", ".pt", ".tmp"}
EXCLUDE_NAMES = {".DS_Store", "MANIFEST.sha256"}


def iter_files(root: Path):
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        if any(part in EXCLUDE_DIRS for part in p.relative_to(root).parts):
            continue
        if p.suffix in EXCLUDE_SUFFIX or p.name in EXCLUDE_NAMES:
            continue
        yield p


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("."))
    ap.add_argument("--out", type=Path, default=None, help="목록을 쓸 경로")
    ap.add_argument("--check", type=Path, default=None, help="기존 목록과 대조")
    a = ap.parse_args()
    root = a.root.resolve()

    current = {}
    total = 0
    for p in iter_files(root):
        rel = p.relative_to(root).as_posix()
        current[rel] = sha256(p)
        total += p.stat().st_size

    if a.check:
        want = {}
        for line in a.check.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            h, _, rel = line.partition("  ")
            want[rel] = h
        missing = sorted(set(want) - set(current))
        added = sorted(set(current) - set(want))
        changed = sorted(r for r in set(want) & set(current) if want[r] != current[r])
        for label, items in (("없어짐", missing), ("추가됨", added), ("내용 변경", changed)):
            if items:
                print(f"{label} {len(items)}개:")
                for r in items[:20]:
                    print(f"  {r}")
                if len(items) > 20:
                    print(f"  ... 그 외 {len(items) - 20}개")
        if missing or added or changed:
            return 1
        print(f"일치: 파일 {len(current)}개")
        return 0

    lines = [f"# sha256 manifest — 파일 {len(current)}개, 합계 "
             f"{total / 2**20:.2f} MB",
             "# 대조: python src/hash_manifest.py --root . --check MANIFEST.sha256"]
    lines += [f"{h}  {rel}" for rel, h in current.items()]
    text = "\n".join(lines) + "\n"
    if a.out:
        a.out.write_text(text, encoding="utf-8")
        print(f"파일 {len(current)}개, 합계 {total / 2**20:.2f} MB → {a.out}")
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
