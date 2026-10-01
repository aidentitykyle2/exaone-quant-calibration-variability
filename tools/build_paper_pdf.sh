#!/bin/bash
# build_paper_pdf.sh — paper/*.md 에서 열람용 PDF를 만든다
#
# 학술대회 제출본은 이 PDF가 아니라 공식 양식(.hwp/.docx)으로 조판한 파일이다.
# 여기서 만드는 것은 저장소 방문자가 본문을 바로 읽기 위한 사본이다.
#
# 경로: markdown -> (수식 유니코드 치환) -> HTML -> wkhtmltopdf
#
# xelatex 경로를 쓰지 않는 이유: 최소 TeX 설치에 lmodern.sty 가 없는 환경이
# 흔하고, 한글 폰트 지정까지 맞추려면 TeX 패키지를 추가로 받아야 한다.
# wkhtmltopdf + Noto CJK 조합은 추가 설치 없이 동작한다.
#
# 필요한 것: pandoc, wkhtmltopdf, Noto Serif/Sans CJK KR
#   Debian/Ubuntu: apt-get install pandoc wkhtmltopdf fonts-noto-cjk
#
# 사용법:
#   bash tools/build_paper_pdf.sh                 # paper_short.md 만
#   bash tools/build_paper_pdf.sh --all           # paper.md 까지

set -euo pipefail
cd "$(dirname "$0")/.."

for b in pandoc wkhtmltopdf; do
  command -v "${b}" >/dev/null || { echo "오류: ${b} 가 없다"; exit 1; }
done

FONT_SERIF="Noto Serif CJK KR"
FONT_SANS="Noto Sans CJK KR"
if ! fc-list : family | tr ',' '\n' | grep -qx "${FONT_SERIF}"; then
  echo "경고: '${FONT_SERIF}' 가 없다. 한글이 깨질 수 있다 (fonts-noto-cjk 설치)"
fi

TMP="$(mktemp -d)"
trap 'rm -rf "${TMP}"' EXIT

cat > "${TMP}/style.css" <<CSS
@page { size: A4; margin: 22mm 20mm; }
body { font-family: "${FONT_SERIF}", serif; font-size: 10.5pt; line-height: 1.65;
       text-align: justify; color: #111; }
h1 { font-family: "${FONT_SANS}", sans-serif; font-size: 16pt; line-height: 1.35;
     margin: 0 0 .2em; }
h2 { font-family: "${FONT_SANS}", sans-serif; font-size: 12.5pt;
     margin: 1.6em 0 .5em; border-bottom: 1px solid #ccc; padding-bottom: .15em; }
h3 { font-family: "${FONT_SANS}", sans-serif; font-size: 11pt; margin: 1.2em 0 .4em; }
p { margin: 0 0 .65em; }
table { border-collapse: collapse; width: 100%; font-size: 9.5pt; margin: .8em 0; }
th, td { border: 1px solid #bbb; padding: 3px 6px; }
th { background: #f2f2f2; font-family: "${FONT_SANS}", sans-serif; }
code { font-family: "Noto Sans Mono CJK KR", monospace; font-size: 9.5pt; }
hr { border: 0; border-top: 1px solid #ddd; margin: 1.4em 0; }
.eq { text-align: center; font-style: italic; margin: .9em 0; }
CSS

# 수식을 유니코드 평문으로 바꾼다. wkhtmltopdf 의 렌더러가 MathML을 제대로
# 그리지 못하므로, 원본 .md 는 LaTeX 수식을 그대로 두고 빌드 사본만 바꾼다.
unmath () {
  python3 - "$1" "$2" <<'PY'
import re, sys, pathlib
src, dst = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
s = src.read_text(encoding="utf-8")

block = {
  r"$$D(c, B) = \frac{Acc_{BF16}(B) - Acc_{c}(B)}{Acc_{BF16}(B)} \times 100$$":
    "<div class=\"eq\">D(c, B) = 100 × ( Acc<sub>BF16</sub>(B) − Acc<sub>c</sub>(B) ) / Acc<sub>BF16</sub>(B)</div>",
  r"$$L_{KO} = D(\text{en}, B_{KO}) - D(\text{ko}, B_{KO}), \qquad L_{EN} = D(\text{en}, B_{EN}) - D(\text{ko}, B_{EN})$$":
    "<div class=\"eq\">L<sub>KO</sub> = D(en, B<sub>KO</sub>) − D(ko, B<sub>KO</sub>) &nbsp;&nbsp;&nbsp; "
    "L<sub>EN</sub> = D(en, B<sub>EN</sub>) − D(ko, B<sub>EN</sub>)</div>",
  r"$$X = L_{KO} - L_{EN}$$": "<div class=\"eq\">X = L<sub>KO</sub> − L<sub>EN</sub></div>",
}
for k, v in block.items():
    s = s.replace(k, v)

# 남은 인라인 수식을 거칠게 평문화한다.
def inline(m):
    t = m.group(1)
    t = re.sub(r"\\binom\{(\d+)\}\{(\d+)\}", r"C(\1,\2)", t)
    t = re.sub(r"\\text\{([^}]*)\}", r"\1", t)
    t = re.sub(r"_\{([^}]*)\}", lambda x: "_" + x.group(1), t)
    t = t.replace("\\qquad", "  ").replace("\\times", "×").replace("\\,", " ")
    t = re.sub(r"\\[a-zA-Z]+", "", t)
    return t.replace("{", "").replace("}", "").strip()

s = re.sub(r"\$\$(.+?)\$\$", lambda m: inline(m), s, flags=re.S)
s = re.sub(r"\$([^$\n]+?)\$", inline, s)
dst.write_text(s, encoding="utf-8")
PY
}

build () {           # $1=입력 md  $2=출력 pdf
  local md="$1" pdf="$2" base
  base="$(basename "${md}" .md)"
  unmath "${md}" "${TMP}/${base}.md"
  pandoc "${TMP}/${base}.md" -f markdown+pipe_tables -t html5 -s \
    --metadata title="" --css "${TMP}/style.css" -o "${TMP}/${base}.html"
  wkhtmltopdf --quiet --enable-local-file-access \
    --page-size A4 --margin-top 0 --margin-bottom 0 \
    --margin-left 0 --margin-right 0 \
    "${TMP}/${base}.html" "${pdf}"
  echo "  생성: ${pdf}  ($(du -h "${pdf}" | cut -f1))"
}

build paper/paper_short.md paper/paper_short.pdf
if [ "${1:-}" = "--all" ]; then
  build paper/paper.md paper/paper_full.pdf
fi
echo "완료"
