# 데이터

이 디렉터리는 저장소를 클론한 직후에는 비어 있다. 실험 서버에서
`scripts/make_release.sh`를 실행해 채운다.

## 여기에 없는 것과 그 이유

| | 이유 | 어떻게 얻는가 |
|---|---|---|
| 양자화 체크포인트 | 조건당 5~18 GB, 전체 수백 GB | 고정 리비전 + `calib_manifests/`의 표본 목록으로 재생성 |
| `--log_samples` 원본 기록 | 실행당 수백 MB, 전체 수 GB | `per_item_scores/`로 손실 없이 요약 (아래) |
| 캘리브레이션 표본의 본문 텍스트 | 원 데이터셋 라이선스 | 기록된 원본 행 번호로 KoAlpaca-v1.1a / Alpaca에서 복원 |

## calib_manifests/

조건(언어, 표본 수, 시드)별로 두 파일이다.

- `calib_<lang>_n<N>_s<seed>.jsonl` — 표본별 원본 행 번호(`source_id`)와 본문
  sha1. 순서가 실제 사용 순서다.
- `calib_<lang>_n<N>_s<seed>.meta.json` — 집합 sha256, 선택 규칙, `datasets`
  버전, 기록 시점(`origin: runtime` = 양자화와 동시 기록, `reconstructed` =
  사후 재구성).

조건별 집합 sha256이 모두 상이함을 확인한 것이 "표본을 실제로 바꿨다"는 근거다.
표본을 복원하려면:

```python
from datasets import load_dataset
ds = load_dataset("beomi/KoAlpaca-v1.1a", split="train")
rows = [json.loads(l)["source_id"] for l in open("calib_ko_n256_s1.jsonl")]
texts = [" ".join([ds[i]["instruction"], ds[i]["output"]]).strip() for i in rows]
```

기록된 sha1과 대조해 복원이 정확한지 확인할 수 있다.

## per_item_scores/

모델·평가언어별 `.npz` 하나와 `.index.json` 하나다. 실행·과제별로 세 배열을
담는다.

| 배열 | 내용 |
|---|---|
| `<run>__<task>__first` | 동률이면 첫 선택 규칙의 정답 여부 (int8) |
| `<run>__<task>__hit` | 정답이 동률 집합에 포함되는지 (int8) |
| `<run>__<task>__ntied` | 동률 선택지 개수 (int8) |
| `<run>__<task>__acc_norm` | 길이 정규화 과제는 기록된 값 그대로 (float32) |
| `order__<task>` | 표준 문항 순서 (`subtask:doc_id`) |

동률 분할 점수는 `hit / ntied`로 **정확히** 복원된다. 따라서 논문의 두 채점
규칙(4.7절의 1.87배 대비), 두 단계 부트스트랩(4.1절), 라벨 재배정 순열
검정(4.2절)을 원본 기록 없이 재현할 수 있다.

`index.json`의 `order_sha256`은 실행 간 문항 순서가 같은지 보증한다. 다르면
대응 부트스트랩의 전제가 깨지므로, 내보내기 시점에 경고하고 종료코드 2를
반환한다.

```python
import numpy as np
z = np.load("EXAONE-3.5-7.8B-Instruct_ko.npz", allow_pickle=True)
ts = z["gptq_ko_s1__kmmlu__hit"] / z["gptq_ko_s1__kmmlu__ntied"]
print(ts.mean())          # rescore_all.py 의 tie_split 값과 일치해야 한다
```

## results/

실행별 `results.json`(환경 정보, 효율성 측정, 오류 기록)과 집계 CSV다.
`environment` 필드가 논문 부록 A의 재현 정보 원천이다.
