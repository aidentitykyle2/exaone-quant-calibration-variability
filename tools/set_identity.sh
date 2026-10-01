#!/bin/bash
# set_identity.sh — 저장소의 저자·계정 자리표시자를 실제 값으로 한 번에 채운다
#
# 저장소에는 확정할 수 없는 값 여섯 개가 @@이름@@ 형태로 남아 있다.
# 이 스크립트가 AUTHORS, CITATION.cff, LICENSE, paper/LICENSE, README.md를
# 한꺼번에 고친다. 실행이 종료코드 0으로 끝나면 업로드 가능 상태다.
#
# 사용법:
#   bash tools/set_identity.sh \
#     --author1-ko "박기술"   --given1 "Kisul" \
#     --author2-ko "정남부"   --given2 "Nambu" \
#     --email2 "co.author@example.ac.kr" \
#     --github "myaccount"
#
#   --author1-ko  제1저자 한글 성명 (예: 박기술)
#   --given1      제1저자 영문 이름 — 성(Park)은 고정, 이름만 (예: Kisul)
#   --author2-ko  교신저자 한글 성명
#   --given2      교신저자 영문 이름 — 성(Jung)은 고정
#   --email2      교신저자 이메일
#   --github      GitHub 계정명 (저장소 URL에 쓰인다)
#
# 제1저자 이메일(aidentitykyle@gmail.com)과 두 소속은 이미 기입되어 있다.
# 바꾸려면 아래 FIXED_* 값을 수정하거나 치환 후 직접 편집한다.

set -euo pipefail
cd "$(dirname "$0")/.."

A1="" G1="" A2="" G2="" E2="" GH=""
while [ $# -gt 0 ]; do
  case "$1" in
    --author1-ko) A1="$2"; shift 2 ;;
    --given1)     G1="$2"; shift 2 ;;
    --author2-ko) A2="$2"; shift 2 ;;
    --given2)     G2="$2"; shift 2 ;;
    --email2)     E2="$2"; shift 2 ;;
    --github)     GH="$2"; shift 2 ;;
    -h|--help)    sed -n '2,25p' "$0"; exit 0 ;;
    *) echo "알 수 없는 인자: $1" >&2; exit 1 ;;
  esac
done

missing=()
[ -z "${A1}" ] && missing+=("--author1-ko")
[ -z "${G1}" ] && missing+=("--given1")
[ -z "${A2}" ] && missing+=("--author2-ko")
[ -z "${G2}" ] && missing+=("--given2")
[ -z "${E2}" ] && missing+=("--email2")
[ -z "${GH}" ] && missing+=("--github")
if [ ${#missing[@]} -gt 0 ]; then
  echo "오류: 다음 인자가 비어 있다 — ${missing[*]}" >&2
  echo '      bash tools/set_identity.sh --help 로 사용법을 본다.' >&2
  exit 1
fi

# GitHub 계정명에 쓸 수 없는 문자가 섞이면 URL이 깨진다.
case "${GH}" in
  *[!A-Za-z0-9-]*) echo "오류: GitHub 계정명에 영문·숫자·하이픈만 쓸 수 있다: ${GH}" >&2; exit 1 ;;
esac
case "${E2}" in
  *@*.*) : ;;
  *) echo "오류: 이메일 형식이 아니다: ${E2}" >&2; exit 1 ;;
esac

TARGETS="AUTHORS CITATION.cff LICENSE paper/LICENSE README.md"

# sed 구분자로 쓰지 않는 문자를 쓰기 위해 | 를 쓴다. 값에 | 가 들어가면 거부한다.
for v in "${A1}" "${G1}" "${A2}" "${G2}" "${E2}" "${GH}"; do
  case "${v}" in *"|"*) echo "오류: 값에 | 를 쓸 수 없다: ${v}" >&2; exit 1 ;; esac
done

for f in ${TARGETS}; do
  [ -f "${f}" ] || { echo "경고: ${f} 가 없다 (건너뜀)"; continue; }
  sed -i \
    -e "s|@@AUTHOR1_KO@@|${A1}|g" \
    -e "s|@@GIVEN1@@|${G1}|g" \
    -e "s|@@AUTHOR2_KO@@|${A2}|g" \
    -e "s|@@GIVEN2@@|${G2}|g" \
    -e "s|@@EMAIL2@@|${E2}|g" \
    -e "s|@@GITHUB@@|${GH}|g" \
    "${f}"
  echo "  채움: ${f}"
done

echo ""
# 점검 범위는 치환 대상 5개 파일뿐이다. docs/SUBMISSION.md 와 이 스크립트는
# 토큰 이름을 설명으로 적고 있으므로 저장소 전체를 긁으면 항상 걸린다.
if grep -lE '@@[A-Z0-9_]+@@' ${TARGETS} >/dev/null 2>&1; then
  echo "남은 자리표시자:"
  grep -nE '@@[A-Z0-9_]+@@' ${TARGETS} | sed 's/^/  /'
  exit 1
fi

echo "치환 대상 파일에 남은 자리표시자가 없다. 다음으로 해시 목록을 갱신한다:"
echo "  python src/hash_manifest.py --root . --out MANIFEST.sha256"
