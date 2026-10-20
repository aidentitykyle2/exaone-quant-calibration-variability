# 데이터

이 디렉터리는 논문 「4비트 LLM 양자화의 언어 비대칭 효과와 신뢰성」의 공개용
데이터다. 실험 서버에서 `scripts/make_release.sh`로 추출한 뒤 논문 범위만
남기도록 정리했다.

## 수록 내용

파일 165개다.

- `calib_manifests/` 표본 목록 78개와 메타데이터 78개
  (EXAONE-3.5-7.8B-Instruct 66쌍, EXAONE-3.5-32B-Instruct 12쌍)
- `per_item_scores/` 문항별 점수 `.npz` 4개와 인덱스 `.index.json` 4개
  (두 모델 × KMMLU, MMLU)
- 이 README 1개

논문에 쓴 조건만 담았다.

- 7.8B GPTQ: 표본 수 128 / 256 / 512 / 1,024
- 7.8B AWQ: 표본 수 256
- 32B GPTQ: 표본 수 256
- 캘리브레이션: 한국어 KoAlpaca-v1.1a(`ko`), 영어 Alpaca(`en_instruct`)
- 평가: KMMLU, MMLU

실험 서버에는 논문에 쓰지 않은 조건(다른 모델, 다른 캘리브레이션 자료, 시험
실행, 다른 벤치마크)의 기록도 있으나 공개본에서는 제외했다. 따라서
`make_release.sh`만 다시 실행하면 이 공개본과 같은 파일 구성이 나오지 않는다.

모델의 고정 리비전과 주요 실행 환경은 저장소 루트 `README.md`의 "환경",
"모델 리비전 고정" 절을 참고한다. 패키지 설치 목록은 `requirements.txt`에
있다.

## 여기에 없는 것과 그 이유

| | 이유 | 어떻게 얻는가 |
|---|---|---|
| 양자화 체크포인트 | 조건당 5~18 GB, 전체 수백 GB | 고정 리비전과 `calib_manifests/`의 표본 목록으로 다시 양자화한다. 단, 투입 순서는 기록되지 않았다(아래) |
| `--log_samples` 원본 평가 로그 | 실행당 수백 MB, 전체 수 GB | 포함하지 않는다. 두 채점 규칙의 정확도를 계산할 수 있는 문항별 점수를 `per_item_scores/`에 제공한다. 선택지별 로그우도 등 원본 로그의 모든 정보가 보존되는 것은 아니다 |
| 캘리브레이션 표본의 본문 텍스트 | 본문은 재배포하지 않고 원본 식별자와 해시만 제공한다 | 원 데이터셋의 이용 조건에 따라 본문을 확보하여 기록된 원본 행 번호로 복원한다(아래) |

## calib_manifests/

모델별 폴더 아래, 조건(언어, 표본 수, 시드)별로 두 파일이다.

- `calib_<lang>_n<N>_s<seed>.jsonl` — 표본별 원본 행 번호(`source_id`)와 본문
  sha1. 본문 텍스트는 들어 있지 않다.
- `calib_<lang>_n<N>_s<seed>.meta.json` — 목록 해시(`set_sha256`), 선택 규칙,
  `datasets` 버전, 기록 시점(`origin`). 공개본에는 본문이 없으므로
  `text_included`는 `false`다.

### 행 순서와 투입 순서

표본 목록의 행 순서는 표본을 선택하여 기록한 순서다. 양자화 라이브러리
(llmcompressor)는 기본 설정에서 투입 순서를 실행마다 다시 섞으며 그 난수를
고정하지 않으므로, 행 순서는 실제 처리 순서를 뜻하지 않는다. 같은 목록으로
다시 양자화해도 논문과 같은 가중치가 나온다는 보장은 없다.

### 목록 해시

`set_sha256`은 표본 본문을 기록 순서대로 결합하여 계산한 SHA-256이다.

각 모델 안에서 서로 다른 언어·표본 수·시드 조합의 목록 해시는 서로 다르다.
표본은 언어, 표본 수, 시드로만 정해지므로, 이 세 가지가 모두 같은 조건에서는
모델·기법 간에 표본 목록을 공유한다.

- 7.8B: 목록 66개, 서로 다른 해시 66개
- 32B: 목록 12개, 서로 다른 해시 12개
- 두 모델을 합치면 목록 78개, 서로 다른 해시 66개

32B의 목록 12개는 대응하는 7.8B 목록과 동일하며, 7.8B AWQ도 같은 선택 조건의
GPTQ 목록을 사용한다.

### 기록 출처

- `origin: runtime` — 양자화와 동시에 기록 (32B 12개, 7.8B 30개)
- `origin: reconstructed` — 실행 당시와 같은 선택 규칙으로 사후에 다시 추출
  (7.8B 36개)

재구성 규칙의 확인: `runtime` 기록 6개(32B 3개, 7.8B 3개)를 같은 규칙으로 다시
추출해 목록 해시가 모두 일치했다. 또 표본 수 256개의 같은 언어 조건에서 7.8B
시드 1, 2번(`reconstructed`)은 32B 시드 1, 2번(`runtime`)과 목록 해시가 같다. `runtime` 기록이 없는 조건은 직접
대조할 원본이 없다.

### 표본 복원

실험 당시 사용한 데이터셋과 동일한 버전 또는 보존된 캐시를 사용해야 한다.
아래 `load_dataset` 호출은 리비전을 지정하지 않으므로, 현재 배포본이 실험
당시와 같다는 것을 보장하지 않는다. 각 `meta.json`의 원본 출처 항목
(`cache_file`)에는 그 기록을 만들 때 사용한 데이터셋 캐시 파일 경로가 있다.
`runtime` 기록은 양자화 실행 시점, `reconstructed` 기록은 재구성 시점의
경로다. 복원한 본문의
SHA-1을 목록에 기록된 값과 대조한다.

```python
import json
from datasets import load_dataset

# 한국어: beomi/KoAlpaca-v1.1a, 영어: tatsu-lab/alpaca (필드 처리는 같다)
ds = load_dataset("beomi/KoAlpaca-v1.1a", split="train")
rows = [json.loads(l)["source_id"] for l in open("calib_ko_n256_s1.jsonl")]
texts = [" ".join([ds[i]["instruction"], ds[i]["output"]]).strip() for i in rows]
```

기록된 sha1과 대조해 복원이 정확한지 확인할 수 있다.

## per_item_scores/

모델·평가언어별 `.npz` 하나와 `.index.json` 하나다. 실행별로 세 배열을 담는다.

| 배열 | 내용 |
|---|---|
| `<run>__<task>__first` | 동률이면 첫 선택 규칙의 정답 여부 (int8) |
| `<run>__<task>__hit` | 정답이 동률 집합에 포함되는지 (int8) |
| `<run>__<task>__ntied` | 동률 선택지 개수 (int8) |
| `order__<task>` | 문항 식별자 순서 (`subtask:doc_id`) |

동률 분할 문항 점수는 `hit / ntied`로 복원된다. 이 배열은 두 채점 규칙(첫
선택, 동률 분할)의 정확도, X, 정확도 기반 순열 검정과 부트스트랩에 필요한 문항
점수를 제공한다. 선택지별 로그우도는 담지 않았으므로, 그것이 필요한 보조
지표(PPL, choice-KL 등)는 이 배열만으로 복원되지 않는다.

저장소의 기존 분석 스크립트가 NPZ를 직접 입력받는 것은 아니므로, 입력 변환이나
별도 분석 코드가 필요하다. 논문과 대조할 때는 채점 규칙, 시드 범위, 반올림
시점과 재표집 설정을 맞춰야 한다.

내보내기 코드는 과제별로 실행 간 공통 문항 식별자를 정렬하고, 각 실행의 점수
배열을 그 순서에 맞춘다. `order_sha256`은 이렇게 정한 기준 문항 식별자 순서의
해시다. 실행 간 문항 집합의 차이를 감지하면 경고와 종료코드 2를 반환하며,
공개본 추출에서는 해당 경고가 나오지 않았다. 정답 선택지 번호를 판별하지 못해
제외한 문항 수는 과제별로 `index.json`의 `dropped_items`에 기록되며, 공개본에는
해당 항목이 없다. 점수 기록(`filtered_resps`)이 없는 문항은 별도 기록 없이
제외된다. 이 해시는 원본 평가 로그의 기록 순서나
문제 본문·선택지·프롬프트가 실행 간 동일했음을 보증하는 값은 아니다.

```python
import numpy as np
z = np.load("EXAONE-3.5-7.8B-Instruct_ko.npz", allow_pickle=False)
ts = z["gptq_ko_s1__kmmlu__hit"] / z["gptq_ko_s1__kmmlu__ntied"]
print(ts.mean())          # rescore_all.py 의 tie_split 값과 일치해야 한다
```

## 무결성 확인

저장소 루트의 `MANIFEST.sha256`에 저장소 전체 파일의 sha256이 있다. 이
디렉터리 외 저장소 파일의 항목도 들어 있으므로, 전체 확인은 저장소 루트에서
`python src/hash_manifest.py --root . --check MANIFEST.sha256`로 한다.
