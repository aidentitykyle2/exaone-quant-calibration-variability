#!/bin/bash
# make_release.sh — 실험 서버의 산출물에서 저장소용 데이터만 추출한다
#
# 실험 디렉터리에는 체크포인트 수백 GB와 lm-eval 원본 기록 수 GB가 있다.
# 그중 재현에 실제로 필요한 것만 골라 저장소의 data/ 아래로 옮긴다.
#
#   calib_manifests/   조건별 표본의 원본 행 번호·sha1·집합 sha256 (본문 텍스트 제외)
#   per_item_scores/   문항별 first / hit / ntied 배열 (동률 분할 점수를 정확히 복원)
#   results/           results.json 과 집계 CSV
#
# 크기 상한을 넘는 파일은 복사하지 않고 경고한다. 저장소에 수백 MB 파일이
# 실수로 들어가는 것을 막는 장치다.
#
# 사용법:
#   bash scripts/make_release.sh <실험_루트> <저장소_루트>
#   bash scripts/make_release.sh /path/to/quant_exp .
#
# 실험 루트 아래에 quant_experiments/<모델명>/ 이 있다고 가정한다.

set -u
EXP="${1:?실험 루트를 지정해야 한다. 예: /path/to/quant_exp}"
REPO="${2:-.}"
MAX_MB="${MAX_MB:-40}"          # 파일 하나의 상한 (MB)
MAX_TOTAL_MB="${MAX_TOTAL_MB:-150}"

[ -d "${EXP}/quant_experiments" ] || { echo "오류: ${EXP}/quant_experiments 가 없다"; exit 1; }
[ -d "${REPO}/src" ] || { echo "오류: ${REPO} 가 이 저장소의 루트가 아니다"; exit 1; }
[ -z "${VIRTUAL_ENV:-}" ] && echo "경고: venv 미활성화 — numpy/datasets 가 필요하다"

D="${REPO}/data"
mkdir -p "${D}/calib_manifests" "${D}/per_item_scores" "${D}/results"
skipped=0

copy_guarded () {   # $1=원본 $2=대상
  local src="$1" dst="$2" kb
  kb=$(du -k "${src}" | cut -f1)
  if [ "${kb}" -gt $((MAX_MB * 1024)) ]; then
    echo "  [건너뜀] $(basename "${src}") — $((kb / 1024)) MB > ${MAX_MB} MB 상한"
    skipped=$((skipped + 1))
    return 1
  fi
  cp "${src}" "${dst}" && return 0
}

for MD in "${EXP}"/quant_experiments/*/; do
  MODEL="$(basename "${MD}")"
  echo "######## ${MODEL} ########"

  # --- 1. 캘리브레이션 표본 목록 (본문 텍스트 제외)
  echo "[1/3] 캘리브레이션 표본 목록"
  if [ ! -d "${MD}/calib_manifests" ]; then
    echo "  기록이 없다 — export_calib_manifests.py 로 사후 재구성을 시도한다"
    ( cd "${EXP}" && python "${REPO}/src/export_calib_manifests.py" \
        --base "quant_experiments/${MODEL}" ) || echo "  재구성 실패 (건너뜀)"
  fi
  if [ -d "${MD}/calib_manifests" ]; then
    OUT="${D}/calib_manifests/${MODEL}"
    mkdir -p "${OUT}"
    # meta.json 은 그대로, jsonl 은 본문 텍스트를 떼고 옮긴다.
    cp "${MD}"/calib_manifests/*.meta.json "${OUT}/" 2>/dev/null
    python - "${MD}/calib_manifests" "${OUT}" <<'PY'
import json, sys, pathlib, hashlib
src, dst = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
n = 0
for p in sorted(src.glob("*.jsonl")):
    rows = []
    for line in p.open(encoding="utf-8"):
        r = json.loads(line)
        t = r.get("text", "")
        rows.append({"lang": r.get("lang"), "source": r.get("source"),
                     "source_id": r.get("source_id"),
                     "sha1": r.get("sha1") or hashlib.sha1(t.encode()).hexdigest(),
                     "chars": len(t)})
    (dst / p.name).write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
        encoding="utf-8")
    n += 1
print(f"  표본 목록 {n}개 (본문 텍스트 제외, 원본 행 번호와 sha1 보존)")
PY
  fi

  # --- 2. 문항별 점수 배열
  echo "[2/3] 문항별 점수 배열"
  if [ -d "${MD}/seed_eval" ]; then
    ( cd "${EXP}" && python "${REPO}/src/export_item_scores.py" \
        --base "quant_experiments/${MODEL}" \
        --out "${D}/per_item_scores" )
    rc=$?
    [ ${rc} -eq 2 ] && echo "  경고: 실행 간 문항 집합이 어긋난 과제가 있다 (위 출력 확인)"
  else
    echo "  seed_eval/ 이 없다 — 건너뜀"
  fi

  # --- 3. results.json 과 집계 CSV
  echo "[3/3] 결과 파일"
  OUT="${D}/results/${MODEL}"
  mkdir -p "${OUT}"
  for f in "${MD}"/results.json "${MD}"/*.csv "${MD}"/rescore_*.csv "${MD}"/mechanism/*.json; do
    [ -f "${f}" ] || continue
    copy_guarded "${f}" "${OUT}/"
  done
  ls "${OUT}" >/dev/null 2>&1 && echo "  $(ls "${OUT}" | wc -l)개 복사"
done

echo ""
echo "######## 요약 ########"
TOTAL_KB=$(du -sk "${D}" | cut -f1)
echo "data/ 합계: $((TOTAL_KB / 1024)) MB"
find "${D}" -type f -size +$((MAX_MB * 1024))k -printf "  큰 파일: %p (%kKB)\n" 2>/dev/null
if [ $((TOTAL_KB / 1024)) -gt "${MAX_TOTAL_MB}" ]; then
  echo "경고: 합계가 ${MAX_TOTAL_MB} MB를 넘는다. git 저장소로는 과하다."
  echo "      per_item_scores/ 를 GitHub Release 자산이나 Zenodo로 분리할 것."
fi
[ "${skipped}" -gt 0 ] && echo "상한 초과로 건너뛴 파일: ${skipped}개"
echo ""
echo "다음 단계:"
echo "  python ${REPO}/src/hash_manifest.py --root ${REPO} --out ${REPO}/MANIFEST.sha256"
echo "  git add -A && git status        # .gitignore 가 체크포인트를 막는지 확인"
