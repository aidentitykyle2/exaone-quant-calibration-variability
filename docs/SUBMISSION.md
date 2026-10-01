# 제출·공개 점검

순서대로 실행하면 업로드 가능 상태가 된다. 1번만 수동 입력이 필요하고 나머지는
명령 실행과 확인이다.

## 1. 저자 정보 채우기 (필수, 1회)

저장소에는 확정할 수 없는 값 여섯 개가 `@@이름@@` 자리표시자로 남아 있다.
제1저자 이메일과 두 소속, 발표 학회, 연도는 이미 기입되어 있다.

| 자리표시자 | 들어갈 값 |
|---|---|
| `@@AUTHOR1_KO@@` | 제1저자 한글 성명 |
| `@@GIVEN1@@` | 제1저자 영문 이름 (성 `Park`은 고정) |
| `@@AUTHOR2_KO@@` | 교신저자 한글 성명 |
| `@@GIVEN2@@` | 교신저자 영문 이름 (성 `Jung`은 고정) |
| `@@EMAIL2@@` | 교신저자 이메일 |
| `@@GITHUB@@` | GitHub 계정명 |

```bash
bash tools/set_identity.sh \
  --author1-ko "박○○" --given1 "Gildong" \
  --author2-ko "정○○" --given2 "Gildong" \
  --email2 "co.author@example.ac.kr" \
  --github "myaccount"
```

`AUTHORS`, `CITATION.cff`, `LICENSE`, `paper/LICENSE`, `README.md` 다섯 파일이
함께 고쳐진다. 스크립트는 남은 자리표시자가 있으면 종료코드 1로 끝난다. 수동
확인:

```bash
grep -nE '@@[A-Z0-9_]+@@' AUTHORS CITATION.cff LICENSE paper/LICENSE README.md
echo "종료코드 $? (1이면 일치 없음 = 정상)"
```

`docs/SUBMISSION.md`와 `tools/set_identity.sh`는 토큰 이름을 설명으로 적고 있어
의도적으로 남는다. 점검 범위를 위 다섯 파일로 한정하는 이유다.

소속이나 제1저자 이메일을 바꾸려면 치환 후 `AUTHORS`와 `CITATION.cff`를 직접
편집한다.

## 2. 데이터 채우기 (실험 서버에서)

클론 직후 `data/`는 비어 있다. 실험 서버에서 산출물을 추출해 채운다.

```bash
bash scripts/make_release.sh /path/to/quant_exp .
```

체크포인트와 lm-eval 원본 기록은 복사되지 않는다. 캘리브레이션 표본 목록,
문항별 점수 배열, `results.json`만 들어온다. 합계가 150 MB를 넘으면 스크립트가
경고하며, 그 경우 `data/per_item_scores/`를 GitHub Release 자산이나 Zenodo로
분리하고 `data/README.md`에 다운로드 위치를 적는다. Zenodo를 쓰면 DOI가 생겨
논문에서 인용할 수 있다.

데이터 없이 코드만 먼저 공개해도 된다. `data/*/.gitkeep`이 디렉터리 구조를
유지하므로 `data/README.md`가 "아직 비어 있음"을 설명한다.

## 3. 저장소 위생

```bash
# 대용량 파일이 섞이지 않았는지
find . -path ./.git -prune -o -type f -size +5M -print

# 체크포인트·원본 기록이 .gitignore 에 걸리는지
git check-ignore -v quant_experiments/x/model.safetensors \
  seed_eval/ko/bf16/samples_a_2026-01-01T00-00-00.jsonl

# 실제 경로·사용자명이 남지 않았는지 (/path/to 로 일반화해 둔 상태를 확인)
grep -rInE '/(home|data|Users)/[a-z0-9_]+' . | grep -v '/path/to' | head -20

# 해시 목록 갱신 — 1번과 2번을 끝낸 뒤에 실행한다
python src/hash_manifest.py --root . --out MANIFEST.sha256
```

마지막 명령은 반드시 다른 모든 수정 뒤에 실행한다. 파일을 고친 뒤 갱신하지
않으면 해시 목록이 저장소 내용과 어긋나 재현성 근거로 쓸 수 없다.

## 4. 코드가 실제로 도는지

깨끗한 환경에서 확인한다. 저장소에 있는 코드가 돌지 않는 것이 재현성 저장소의
가장 흔한 실패다.

```bash
python -m venv /tmp/fresh && source /tmp/fresh/bin/activate
pip install -r requirements.txt

# 전체 파이썬 파일 문법 검사
python -c "import ast,pathlib,sys
bad=[]
for p in pathlib.Path('.').rglob('*.py'):
    if '.git' in p.parts: continue
    try: ast.parse(p.read_text(encoding='utf-8'))
    except SyntaxError as e: bad.append((str(p), e))
print('문법 오류', len(bad)); [print(' ', *b) for b in bad]
sys.exit(1 if bad else 0)"

# 셸 스크립트 문법 검사
for f in $(find scripts future_work tools -name '*.sh'); do bash -n "$f" || echo "문법 오류: $f"; done

# GPU 불필요한 자체 점검
python future_work/robust_selftest.py
```

## 5. 논문 본문과의 대조

- `paper/references.md`의 번호가 `paper_short.md`·`paper.md`의 본문 인용과 같은지
- 부록 A의 리비전 해시가 `README.md`의 표와 같은지
- 부록 A의 라이브러리 버전이 `requirements.txt`와 같은지
- 본문이 언급하는 스크립트 이름이 저장소에 실제로 있는지

```bash
# 참고문헌 번호와 본문 인용 번호가 일치하는지
python -c "
import re,pathlib
refs={int(m) for m in re.findall(r'^\[(\d+)\]', pathlib.Path('paper/references.md').read_text(encoding='utf-8'), re.M)}
for f in ('paper/paper_short.md','paper/paper.md'):
    t=pathlib.Path(f).read_text(encoding='utf-8')
    used={int(m) for m in re.findall(r'\[(\d{1,2})\]', t)}
    print(f, '목록에 없는 인용:', sorted(used-refs) or '없음')
"

# 본문이 언급하는 파일명이 저장소에 있는지
grep -oE '[a-z_0-9]+\.(py|sh|txt)' paper/paper.md | sort -u | while read f; do
  find . -name "$f" -not -path './.git/*' -print -quit | grep -q . || echo "없음: $f"
done
```

## 6. 공개 시점

이 학술대회는 논문집 발표 형식이므로 투고 시점부터 공개해도 무리가 없다. 다만
투고 규정에 심사 중 저장소 공개를 제한하는 조항이 있는지 한 번 확인한다.

**이중 눈가림 심사에 함께 투고하는 경우에만** 심사 기간에 공개하지 않는다. 공개
저장소는 저자 신원을 드러내는 경로가 여럿이다. 저장소 소유자 계정명, 커밋
작성자 이름과 이메일, `CITATION.cff`와 `AUTHORS`의 저작권자, 서버 경로에 남은
사용자명, 커밋 메시지와 이슈다. 그 경우 anonymous.4open.science 등에 비식별
사본을 올려 논문에는 그 링크만 적고, 원본 저장소는 게재 확정 후 공개한다.

```bash
# 익명화가 필요한 경우의 점검 (대괄호 안을 본인 값으로 바꿔 실행)
grep -rIn --exclude-dir=.git -iE '[사용자명]|[호스트명]|[실명]' . | head -40
git log --format='%an <%ae>' | sort -u
```

## 7. 게재 후

- `CITATION.cff`의 `preferred-citation`에 쪽수와 DOI 추가
- `v1.0` 태그와 GitHub Release 생성, `MANIFEST.sha256`을 자산으로 첨부
- Zenodo 연동으로 저장소 자체의 DOI 발급
- 게재지가 저작권 이전을 요구하면 `paper/LICENSE`와 README 라이선스 절을 갱신
- `future_work/`를 실제로 실행하게 되면 별도 브랜치로 분리하고, 이 저장소는
  논문 시점 상태를 태그로 보존
