# 4비트 LLM 양자화에서 측정된 언어 비대칭 효과의 신뢰성

*How Reliable Are Measured Language Asymmetries in 4-Bit LLM Quantization? An Analysis of Calibration Sampling Variability*

논문에 쓰인 코드, 실행 스크립트, 분석 절차, 재현 정보를 담은 저장소다.
새로운 양자화 기법을 제안하지 않는다. 양자화 실험에서 **측정된 효과의 재현성과
귀속 가능성**을 다룬다.

> **업로드 전 1회 실행**
> 저자 성명과 GitHub 계정명은 `@@이름@@` 자리표시자로 두었다. 아래 한 줄로
> `AUTHORS`·`CITATION.cff`·`LICENSE`·`paper/LICENSE`·`README.md`가 함께 채워진다.
>
> ```bash
> bash tools/set_identity.sh --author1-ko "박○○" --given1 "Gildong" \
>   --author2-ko "정○○" --given2 "Gildong" \
>   --email2 "co.author@example.ac.kr" --github "myaccount"
> ```
>
> 그 밖의 점검 항목은 [`docs/SUBMISSION.md`](docs/SUBMISSION.md)에 있다.

---

## 요약

- 캘리브레이션 표본만 바꿔 같은 실험을 반복하자 한국어·영어 손상 격차가
  −2.45 ~ +5.74 %p에 분포하고 18회 중 5회는 방향이 뒤집혔다.
- 캘리브레이션 언어 라벨을 재배정하는 순열 검정에서, **두 평가 언어의 대비를
  함께 취한 지표는 두 규모·두 기법의 세 설정 모두에서 캘리브레이션 언어에
  귀속되었으나, 선행 연구가 주로 보고해 온 한쪽 언어의 단순 효과는 어느
  설정에서도 귀속되지 않았다.**
- 단일 실행(2.7배)과 동률 첫 선택 채점(1.87배)은 독립적으로 효과 크기를
  부풀린다.

전문은 [`paper/paper.md`](paper/paper.md), 학술대회 제출 요약본은
[`paper/paper_short.md`](paper/paper_short.md)·`paper_short.pdf`.
참고문헌 목록과 본문 인용 위치는 [`paper/references.md`](paper/references.md).

## 저장소 구조

```
paper/                논문 본문·요약본·초록·그림·참고문헌 목록
src/                  양자화·평가·분석 파이썬 코드
scripts/              실행 스크립트와 작업 목록 (GPU 큐 기반)
  jobs/               실제로 사용한 작업 목록
  legacy/             초기 단계 스크립트 (논문 결과에 미사용, 이력 보존)
data/                 재현에 필요한 소형 산출물 (아래 "데이터" 절)
future_work/          본 논문 범위 밖 코드 — 계열 확장, 강건 Hessian 집계
docs/                 실행 절차, 환경 확정 과정, 중간 수정 패치 이력
tools/                저자 정보 채우기, 문서 변환 보조 스크립트
  set_identity.sh     업로드 전 1회 — 자리표시자를 실제 값으로 치환
  build_paper_pdf.sh  paper/*.md → 열람용 PDF (pandoc + wkhtmltopdf)
```

## 환경

| | |
|---|---|
| GPU | NVIDIA H100 NVL 93 GB × 4 (드라이버 560.35.05, CUDA 12.6 상한) |
| torch | 2.10.0+cu126 |
| vLLM | 0.19.0 |
| llmcompressor | 0.10.0.3 |
| compressed-tensors | 0.14.0.1 |
| transformers | 4.57.6 |

이 조합은 임의 선택이 아니다. torch의 CUDA 12.x 지원 종료 시점과 llmcompressor의
transformers 상한(4.57.6), vLLM 최신판의 하한(5.5.3+) 사이에 교집합이 없어
vLLM을 0.19.0으로 낮춰 확보한 조합이다. 확정 과정은
[`docs/process_summary.md`](docs/process_summary.md)에 기록했다.

```bash
python -m venv venv-quant && source venv-quant/bin/activate
pip install -r requirements.txt
```

## 모델 리비전 고정

| 모델 | 리비전 |
|---|---|
| LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct | `0ff6b5ec7c13b049b253a16a889aa269e6b79a94` |
| LGAI-EXAONE/EXAONE-3.5-32B-Instruct | `d6fa88cd8d2c9512b40578bdc44e64909e5a5042` |

두 저장소의 원격 코드(`configuration_exaone.py` 등)가 상류에서 갱신되어
transformers 4.57.6에서 `RopeParameters` import 오류가 발생한다. 리비전을
고정하지 않으면 재현되지 않는다. 다른 모델로 확장할 때 호환 리비전을 찾는
도구는 [`src/find_revision.py`](src/find_revision.py)다.

## 빠른 시작

```bash
source venv-quant/bin/activate
bash scripts/preflight_32b.sh quick          # 구조 점검, GPU 불필요
CUDA_VISIBLE_DEVICES=0 bash scripts/seed_job.sh bf16:ko
CUDA_VISIBLE_DEVICES=0 bash scripts/seed_job.sh sweep:gptq:ko:1:256
```

여러 GPU로 나눠 돌리려면 동적 큐를 쓴다. 각 워커가 `mkdir`로 작업을 원자적으로
선점하므로 워커를 서로 다른 시점에 합류시켜도 된다.

```bash
for g in 0 1 2 3; do
  nohup bash scripts/queue_worker.sh $g scripts/jobs/jobs_78b.txt \
    > logs/gpu$g.txt 2>&1 & sleep 5
done
```

전체 재현 절차와 소요 시간은 [`REPRODUCE.md`](REPRODUCE.md)에 있다.

## 데이터

양자화 체크포인트와 `--log_samples` 원본은 **올리지 않는다.** 체크포인트는
고정 리비전과 표본 목록으로 재생성되고, 원본 기록은 수 GB다. 대신 재현에
실제로 필요한 것만 담는다.

| 위치 | 내용 | 비고 |
|---|---|---|
| `data/calib_manifests/` | 조건별 캘리브레이션 표본의 **원본 행 번호·sha1·집합 sha256**과 선택 규칙 | 본문 텍스트는 원 데이터셋 라이선스를 따르므로 포함하지 않는다. 행 번호로 완전히 복원된다 |
| `data/per_item_scores/` | 실행·과제별 문항 단위 `first` / `hit` / `ntied` 배열 (npz) | 동률 분할 점수는 `hit / ntied`로 **정확히** 복원된다. 두 단계 부트스트랩과 순열 검정을 원본 기록 없이 재현할 수 있다 |
| `data/results/` | 실행별 `results.json`(환경·효율성)과 집계 CSV | |

`data/`를 채우는 절차는 [`scripts/make_release.sh`](scripts/make_release.sh)다.
실험 서버에서 실행하며, 크기 상한을 넘는 파일은 복사하지 않고 경고한다.

```bash
bash scripts/make_release.sh /path/to/quant_exp .
python src/hash_manifest.py --root . --out MANIFEST.sha256
```

## 분석 재현

문항별 점수 배열만 있으면 GPU 없이 논문의 통계 결과가 재현된다.

```bash
python src/rescore_all.py      --base <모델 디렉터리>    # 동률 두 규칙 재채점 (4.7절)
python src/paired_bootstrap.py --base <모델 디렉터리>    # 두 단계 부트스트랩 (4.1절)
python src/permutation_test.py --base <모델 디렉터리>    # 라벨 재배정 순열 검정 (4.2절)
python src/single_run_risk.py  --base <모델 디렉터리>    # 단일 실행 위험 (4.3절)
python src/sd_interval.py      --base <모델 디렉터리>    # 표준편차 구간
python src/item_stability.py   --base <모델 디렉터리>    # 문항 수준 안정성
python src/final_analysis.py   --base <모델 디렉터리>    # 종합 표
python src/make_figure1.py     --help                    # 그림 1
```

## 핵심 정의

$$D(c, B) = \frac{Acc_{BF16}(B) - Acc_{c}(B)}{Acc_{BF16}(B)} \times 100$$

$$L_{KO} = D(\mathrm{en}, B_{KO}) - D(\mathrm{ko}, B_{KO}), \quad
L_{EN} = D(\mathrm{en}, B_{EN}) - D(\mathrm{ko}, B_{EN}), \quad
X = L_{KO} - L_{EN}$$

`c`는 캘리브레이션 언어, `B`는 평가 벤치마크다. 가설이 예측하는 방향은
$L_{KO}>0$, $L_{EN}<0$, $X>0$이다.

## 향후 연구 코드

[`future_work/`](future_work/)는 **논문 결과에 사용되지 않았다.** 두 한계에
대응하는 실행 가능한 코드이며, GPU 없이 검증 가능한 부분은 검증해 두었다.

- **모델 계열 확장** — 동일 설계를 다른 계열에 반복. 사전 점검에 대상 모델의
  BF16 한국어 정확도가 우연 수준이면 지표를 정의할 수 없다는 판단 관문과,
  토크나이저의 한국어 분절 정도가 표본 수 통제를 느슨하게 만드는지 확인하는
  절차를 포함한다.
- **강건 Hessian 집계** — GPTQ의 Hessian 표본평균을 기하중앙값으로 대체한다.
  총 캘리브레이션 예산을 고정하므로 4.3절의 "표본 수를 늘려도 변동이 줄지
  않는다"와 직접 대조된다. 표본평균 집계가 기존 GPTQ와 수치적으로 동일한
  결과를 내는지 확인하는 절차를 포함하며, 통과하지 못하면 본 실험이 실행되지
  않는다.

자세한 내용은 [`future_work/README.md`](future_work/README.md).

## 알려진 제약

- 모델 계열이 EXAONE 3.5 하나다. 규모 확장(7.8B → 32B)은 계열 간 일반화가 아니다.
- "한국어"는 KMMLU, "영어"는 MMLU이므로 언어 효과와 벤치마크 효과가 분리되지 않았다.
- 32B는 6표본으로, 정확도 기준 검출에 필요한 7표본에 못 미친다.
- QuaRot의 R3·R4는 적용하지 않았다. 기술적 제약(고정 리비전의 AttentionInterface
  미지원, `intermediate_size`가 2의 거듭제곱이 아님)과 W4A16에서 항등 연산이라는
  두 근거가 모두 있다. SpinQuant(learnable)는 llmcompressor 0.10.0.3에 미구현이다.
- 실험 중간에 하드코딩 해제 수정이 있었다. 무엇을 언제 고쳤는지는
  [`docs/patches/`](docs/patches/)에 남겨 두었다.

## 인용

```bibtex
@inproceedings{park2026calibration,
  title     = {4비트 LLM 양자화의 언어 비대칭 효과와 신뢰성},
  author    = {Park, JoonOh and Jung, Yuil},
  booktitle = {2026 한국정보기술학회 추계 종합학술대회 논문집},
  year      = {2026}
}
```

## 도구 사용 표기

실험 코드와 분석 스크립트 작성에 Claude Opus 5(Anthropic)의 도움을 받았다.
코드의 동작과 논문에 제시한 모든 수치는 저자가 직접 검증하였다. 검증에 쓴
자체 점검 코드는 [`future_work/robust_selftest.py`](future_work/robust_selftest.py)와
[`future_work/robust_e2e_test.py`](future_work/robust_e2e_test.py)에 있다.

## 라이선스

- 코드(`src/`, `scripts/`, `future_work/`, `tools/`): [MIT](LICENSE)
- 논문 본문과 그림(`paper/`): CC BY 4.0 — [`paper/LICENSE`](paper/LICENSE)
- `data/calib_manifests/`는 원본 텍스트를 포함하지 않는다. KoAlpaca-v1.1a와
  Alpaca의 라이선스는 각 데이터셋 배포처를 따른다.
#   e x a o n e - q u a n t - c a l i b r a t i o n - v a r i a b i l i t y  
 #   e x a o n e - q u a n t - c a l i b r a t i o n - v a r i a b i l i t y  
 