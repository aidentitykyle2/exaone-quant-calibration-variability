# 향후 연구 코드 — (1) 모델 계열 확장, (2) 강건 Hessian 집계

본 단계 논문(EXAONE 3.5-7.8B / 32B)의 범위에는 포함되지 않는다. 논문 8장
향후 연구 절에서 "코드를 공개한다"고 명시할 수 있는 수준으로 작성하고,
GPU 없이 확인 가능한 부분은 실제로 실행해 검증했다.

---

## 0. 두 방향이 각각 어떤 비판에 답하는가

| | (1) 모델 계열 확장 | (2) 강건 Hessian 집계 |
|---|---|---|
| 답하는 비판 | "모델 계열이 하나라 일반화를 알 수 없다" | "진단만 있고 처방이 없다" |
| 논문 위치 | 7장 한계 → 8장 향후 연구 | 6장 논의 → 8장 향후 연구 |
| 기존 코드 수정 | **없음** (환경변수만) | run_quant_stage.py에 패치 1건 |
| 소요 (4 GPU) | 약 20시간 | 약 20시간 + 검증 1~2시간 |
| 실패 위험 | 낮음 (GPTQ는 매핑 불필요) | 중간 (라이브러리 내부 개입) |
| 결과가 음성이면 | 계열 특수성이라는 결론 — 여전히 유용 | "원인은 집계 방식이 아니다" — 4.3절 보강 |

(2)의 논리적 연결: 4.3절에서 표본 수를 128→1024로 늘려도 SD(X)가 줄지
않았다(1.07 / 0.99 / 1.01 / 1.03). 표본 수가 원인이 아니라면 남은 조작 지점은
**통계량을 합치는 방식**이다. 이 코드는 그 가설을 반증 가능한 형태로 만든다.

---

## 1. 모델 계열 확장

### 파일

| 파일 | 역할 |
|---|---|
| `preflight_family.sh` | 3단 사전 점검 (구조 → 측정 가능성 관문 → 실제 양자화) |
| `jobs_family.txt` | BF16 2건 + 18시드 × 2언어 = 38 작업 |
| `run_family.sh` | 4 GPU 동적 큐 실행 |

기존 `seed_job.sh` / `queue_worker.sh`를 **수정하지 않는다**. 이미
`MODEL_ID` / `REVISION` / `BASE` / `GPU_UTIL` 을 환경변수로 받게 만들어 뒀기
때문이고, 논문 결과를 만든 스크립트를 건드리지 않는 것 자체가 재현성 측면의
이점이다.

### 실행 순서

```bash
source venv-quant/bin/activate
MID=Qwen/Qwen2.5-7B-Instruct bash preflight_family.sh quick   # 약 3분
MID=Qwen/Qwen2.5-7B-Instruct bash preflight_family.sh gate    # 약 1~2시간
MID=Qwen/Qwen2.5-7B-Instruct bash preflight_family.sh full    # 약 30~60분
MID=Qwen/Qwen2.5-7B-Instruct nohup bash run_family.sh > run_family.txt 2>&1 & disown
```

### 가장 중요한 지점 — 측정 가능성 관문 (`gate`)

BF16 KMMLU가 우연 수준(4지선다 0.25)에 가까우면, 저하율
`D(c,B) = (Acc_BF16 − Acc_c) / Acc_BF16` 의 분모가 불안정해져 언어 비대칭
지표 X 자체가 의미를 잃는다. 이 관문을 통과하지 못하는 모델에 20 GPU시간을
쓰면 아무 결론도 얻지 못한다. 기본 기준값은 0.33이고 `MIN_KMMLU`로 조절한다.

참고값: EXAONE 3.5-7.8B KMMLU 0.4555, 32B 0.4990.

### 토크나이저 경고를 반드시 볼 것

`preflight_family.sh`의 3단계는 한국어 문자/토큰 비를 출력한다. 이 값이 1.2
미만이면 그 모델이 한국어를 사실상 바이트 단위로 분절한다는 뜻이고, 그러면
**같은 256개 문장이 담는 한국어 정보량이 EXAONE과 크게 달라진다**. "표본 수를
고정했다"는 통제가 느슨해지므로, 표본 수 대신 토큰 수를 맞추는 설계로
바꾸거나 논문에 명시해야 한다. 계열 비교에서 가장 놓치기 쉬운 교란이다.

### 모델 선택이 결론의 성격을 바꾼다

- 한국어 중심 계열(Kanana, Mi:dm, A.X 등) → 본 발견의 **반복 검증**
- 비한국어 중심 계열(Qwen2.5, Llama-3.1) → 사전학습 언어 구성이 조건인지에 대한 **대비**

`preflight_family.sh` 1단계가 저장소 실재·공개 여부·원격 코드 유무를 확인한다.
한국어 중심 계열의 저장소 이름은 본 코드 작성 시점 이후 바뀌었을 수 있으므로
반드시 이 단계로 확인할 것.

기본값을 `Qwen/Qwen2.5-7B-Instruct`로 둔 이유는 transformers 공식 지원이라
원격 코드·리비전 고정 문제가 없고(EXAONE에서 겪은 `RopeParameters` 류의 충돌이
발생하지 않는다), 한국어 성능도 관문을 넘을 가능성이 높기 때문이다.

---

## 2. 강건 Hessian 집계

### 파일

| 파일 | 역할 | 검증 |
|---|---|---|
| `robust_gptq.py` | `RobustGPTQModifier` + `aggregate_hessians()` | 아래 전부 |
| `robust_gptq.patch` | `run_quant_stage.py` 패치 (`--stage gptq_robust`) | 적용 가능 확인 |
| `queue_worker.patch` | `QUEUE_JOB_SCRIPT` 환경변수 1줄 추가 | 적용 가능 확인 |
| `robust_selftest.py` | 집계 수식·PSD·강건성 (GPU 불필요, 수 초) | 통과 |
| `robust_e2e_test.py` | 실제 llmcompressor 파이프라인 통합 (CPU, 수 분) | 통과 |
| `compare_ckpt.py` | 두 체크포인트의 양자화 가중치 직접 비교 | — |
| `robust_job.sh` | `rcheck`(배선 검증) / `rlane`(본 실험) | 문법 확인 |
| `jobs_robust_check.txt` / `jobs_robust.txt` | 검증 2건 / 본 실험 36건 | — |
| `run_robust.sh` | 0→1→2단계 순차 강제 실행 | 문법 확인 |

### 방법

GPTQ는 `H = 2·Σ x_i x_iᵀ` 를 쌓고 압축 시 `H / n`(표본평균)을 쓴다. 표본평균은
소수의 이상 표본에 선형적으로 끌려간다. 캘리브레이션을 K개 부분집합으로 나눠
부분집합별 `M_k = H_k / n_k` 를 구한 뒤 표본평균 대신 강건 추정량으로 합친다.
**총 표본 수는 256으로 고정한다** — 예산을 늘려서 나아진 것이 아니라고 말하려면
이 조건이 반드시 유지되어야 한다.

| 집계 | PSD 보존 | 표본 예산 | 비고 |
|---|---|---|---|
| `mean` | 예 (볼록결합) | 유지 | **기존 GPTQ와 수학적으로 동일** — 대조군·배선 검증용 |
| `geomedian` | 예 (Weiszfeld 볼록결합) | 유지 | **권장 기본값** |
| `trimmed` | 예 (볼록결합) | **감소** ((K−2·trim)/K) | 예산이 줄어 대조가 흐려진다 |
| `median` | **아니오** | 유지 | 원소별 연산 — 주 결과로 쓰지 말 것 |

`geomedian`은 가중치가 비음수이고 합이 1인 볼록결합이라 PSD·대칭이 구조적으로
보존되고, 모든 부분집합을 (가중해서) 쓰므로 표본 예산이 그대로다. 연합학습의
강건 집계(RFA, Pillutla et al.)와 같은 형태다. `trimmed`는 부분집합을 버리므로
예산이 줄어 "같은 예산" 조건이 깨진다 — 보조 비교로만 쓸 것.

### 실제로 확인한 것 (이 대화에서 실행)

`robust_selftest.py` (dim=128, samples=256, K=4):

```
T1 mean == 풀링 (균등 [64,64,64,64]):   상대오차 4.135e-07  통과
T1 mean == 풀링 (불균등 [25,51,76,104]): 상대오차 3.308e-07  통과
T2 K=1 항등 (mean/median/trimmed/geomedian): 모두 0.000e+00  통과
T3 PSD·대칭 (mean/trimmed/geomedian): 최소고유값 모두 양수, 비대칭 0  통과
T4 강건성 (부분집합 1/4을 20배로 오염):
      mean      167.68     <- 표본평균은 이상 표본 하나에 사실상 붕괴한다
      median      0.559
      trimmed     0.476
      geomedian   0.131
T5 분할 수 무영향 (mean, K=2/3/4/8): 최대 상대오차 2.481e-07  통과
T6 median 최소고유값: 투영 전 -3.154e+02 / 투영 후 -2.164e-06
      -> median이 PSD를 실제로 깬다는 실측 근거. 주 결과로 쓰면 안 된다.
```

`robust_e2e_test.py` (무작위 초기화 tiny Llama, llmcompressor 실제 oneshot 경로):

```
기준선 GPTQModifier  vs  RobustGPTQModifier(K=4, agg=mean)
  비교 파라미터 43개 | 최대 상대오차 0.000e+00   <- 완전 일치
  subset_assignment=interleave 로도 0.000e+00   <- 표본 중복·누락 없음
집계 경로가 모두 정상 동작 (기준선과의 평균 상대차)
  geomedian 0.0476 | trimmed 0.0493 | median 0.0448   NaN 없음
```

`mean` 집계가 기준선과 **오차 0으로** 일치한다는 것이 이 코드의 핵심 근거다.
부분집합 분할·훅·집계·QAC 비활성화 배선에 기준선과 다른 요인이 섞여 있지
않다는 뜻이다.

### 구현상 반드시 알아야 할 두 가지

**1) `DISABLE_QAC_MODIFIERS` 등록.** llmcompressor의 파이프라인은
quantization-aware calibration 비활성화 여부를 **클래스 이름 문자열**로
판단한다(`DISABLE_QAC_MODIFIERS = ["GPTQModifier", "AWQModifier",
"AutoRoundModifier"]`). 하위 클래스 이름을 이 목록에 넣지 않으면 QAC가 켜진
상태로 돌아 **조용히 다른 실험이 된다**. `robust_gptq.py` 하단에서 import
시점에 등록한다. 이 실험에서 가장 위험한 함정이다.

**2) 부모 로그의 표본 수가 1.0으로 찍힌다.** 집계 결과를
`self._hessians[module]`에 넣고 `self._num_samples[module] = 1.0`으로 두어
부모의 `H / n`을 항등으로 만들기 때문이다(그 덕에 `quantize_weight`·오프로딩·
`on_finalize` 검증을 그대로 재사용한다). 실제 표본 수와 부분집합 구성은
`[RobustGPTQ] 집계 예시 ...` 로그가 따로 찍는다. 이 줄의 합이 256인지 확인하면
표본 중복·누락 여부를 알 수 있다.

### 메모리

Hessian은 (입력차원)² float32이고, `c_proj`의 입력차원이 `intermediate_size`라
가장 크다.

| | intermediate | Hessian 1개 | K=4 |
|---|---|---|---|
| 7.8B | 14336 | 0.82 GiB | 3.3 GiB |
| 32B | 27392 | 2.80 GiB | 11.2 GiB |

부분집합은 기본적으로 CPU에 둔다(`subset_device="cpu"`). GPU에는 활성 버퍼
하나만 남고, `block` 배정에서는 한 번에 활성 부분집합이 하나뿐이라 전송이
K−1회로 끝난다. 집계 연산도 열 블록 단위로 나눠(`max_chunk_cols=4096`) K개를
한꺼번에 스택하지 않는다. 32B에서 CPU RAM 여유가 부족하면
`max_chunk_cols`를 줄이는 대신 K를 2로 낮추는 편이 낫다.

### 설치와 실행

```bash
cd /path/to/quant_exp
cp /path/to/{robust_gptq.py,robust_selftest.py,robust_e2e_test.py,compare_ckpt.py} .
cp /path/to/{robust_job.sh,run_robust.sh,jobs_robust.txt,jobs_robust_check.txt} .
cp /path/to/{robust_gptq.patch,queue_worker.patch} .

patch -p0 run_quant_stage.py < robust_gptq.patch
patch -p0 queue_worker.sh    < queue_worker.patch
python -c "import ast;ast.parse(open('run_quant_stage.py').read());print('ok')"

source venv-quant/bin/activate
python robust_selftest.py        # 수 초, GPU 불필요
python robust_e2e_test.py        # 수 분, CPU
nohup bash run_robust.sh > run_robust.txt 2>&1 & disown
```

`run_robust.sh`는 0단계(수식 점검) → 1단계(실제 모델 배선 검증, `agg=mean`이
기준선 체크포인트와 같은 가중치를 내는지) → 2단계(본 실험)를 순서대로 강제한다.
1단계가 실패하면 2단계로 넘어가지 않는다. 배선 검증 없이 본 실험을 돌리면
SD(X)가 줄든 늘든 그것이 집계 방식 때문인지 구현 오류 때문인지 구분할 수 없다 —
단일 실행 결과를 믿지 말라고 주장하는 논문이라면 완화 절차에도 같은 기준을
적용해야 한다.

### 판정 기준

기준선 7.8B GPTQ: **SD(X) = 2.00 %p, 18시드 중 5건 부호 반전,
SD_calib/SE_eval = 3.64.**

- SD(X)가 유의하게 줄고 부호 반전이 사라지면 → 완화 가설 채택. 단일 캘리브레이션
  가이드(7장)에 "강건 집계를 쓰라"는 처방을 추가할 수 있다.
- 줄지 않으면 → "변동의 원인은 집계 방식이 아니다". 4.3절(표본 수도 원인이
  아니다)과 합쳐, 변동이 GPTQ의 순차적 반올림 자체에서 온다는 방향을 가리킨다.

두 결과 모두 보고 가치가 있다. 그래서 이 실험은 걸어 볼 만하다.

분석은 기존 도구를 그대로 쓴다. 출력 이름이
`gptq_robust_k4_geomedian_<lang>_s<seed>` 규칙을 따르므로
`rescore_all.py`(tie_split 재채점 — 기준선과 **같은 채점 규칙**을 써야 한다) →
`final_analysis.py` 순서로 돌린다.

---

## 3. 논문 8장에 넣을 문장 (초안)

> 본 연구의 두 한계에 대응하는 코드를 공개한다. 첫째, 모델 계열이 하나라는
> 한계에 대해, 동일 설계를 다른 계열에 반복하는 실행 코드와 사전 점검
> 절차를 제공한다. 사전 점검에는 대상 모델의 BF16 한국어 정확도가 우연
> 수준에 가까울 경우 언어 비대칭 지표를 정의할 수 없다는 판단 관문과,
> 토크나이저의 한국어 분절 정도가 표본 수 통제를 느슨하게 만드는지를
> 확인하는 절차를 포함했다. 둘째, 진단에 대응하는 완화 절차로서, GPTQ의
> Hessian 표본평균을 기하중앙값으로 대체하는 구현을 제공한다. 총
> 캘리브레이션 예산을 고정한 채 집계 방식만 바꾸므로, 4.3절에서 관측한
> "표본 수를 늘려도 변동이 줄지 않는다"는 결과와 직접 대조된다. 이
> 구현은 표본평균 집계가 기존 GPTQ와 수치적으로 동일한 결과를 내는지
> 확인하는 검증 절차를 포함하며, 검증을 통과하지 못하면 본 실험이
> 실행되지 않도록 구성했다.
