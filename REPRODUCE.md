# 재현 절차

세 층으로 나뉜다. 위로 갈수록 싸다. **논문의 통계적 주장은 3층만으로 재현된다.**

| 층 | 무엇을 재현하는가 | 필요한 것 | 소요 |
|---|---|---|---|
| 1. 전체 | 양자화부터 평가까지 | H100 4장, 저장 공간 500 GB+ | 7.8B 약 40시간 + 32B 약 30시간 |
| 2. 평가만 | 기존 체크포인트 재평가 | H100 1장 이상, 체크포인트 | 조건당 약 4.5시간 |
| 3. 분석만 | 논문의 모든 표·그림·검정 | CPU만 | 수 분 |

---

## 3층. 분석만 재현 (권장 시작점)

`data/per_item_scores/`의 문항별 점수 배열만 있으면 GPU가 필요 없다.

```bash
python -m venv venv && source venv/bin/activate
pip install numpy scipy matplotlib        # 분석에는 torch·vLLM이 필요 없다
```

`src/*.py`는 `--base <모델 디렉터리>`로 `seed_eval/` 구조를 읽으므로, 원본
평가 기록이 없는 상태에서는 npz를 직접 읽는다. 배열이 문항 단위이므로 저하율·
비대칭·부트스트랩을 그대로 계산할 수 있다.

```python
import numpy as np, itertools, json
z = np.load("data/per_item_scores/EXAONE-3.5-7.8B-Instruct_ko.npz", allow_pickle=True)
idx = json.load(open("data/per_item_scores/EXAONE-3.5-7.8B-Instruct_ko.index.json"))

def acc(run, task, rule="tie_split"):
    if f"{run}__{task}__acc_norm" in z:
        return float(z[f"{run}__{task}__acc_norm"].mean())
    if rule == "first":
        return float(z[f"{run}__{task}__first"].mean())
    return float((z[f"{run}__{task}__hit"] / z[f"{run}__{task}__ntied"]).mean())

bf16 = acc("bf16", "kmmlu")
for run in idx["runs"]:
    if run == "bf16":
        continue
    print(run, round((bf16 - acc(run, "kmmlu")) / bf16 * 100, 3))   # D(c, B_KO)
```

체크포인트를 가지고 있다면 스크립트를 그대로 쓴다.

```bash
B=quant_experiments/EXAONE-3.5-7.8B-Instruct
python src/rescore_all.py      --base $B     # 4.7절: 두 채점 규칙
python src/paired_bootstrap.py --base $B     # 4.1절: 두 단계 부트스트랩
python src/permutation_test.py --base $B     # 4.2절: 라벨 재배정 순열 검정
python src/single_run_risk.py  --base $B     # 4.3절: SD_calib / SE_eval
python src/sd_interval.py      --base $B     # 표준편차의 구간
python src/item_stability.py   --base $B     # 문항 수준 안정성
python src/m4_full.py          --base $B     # 4.5~4.6절: 네 지표
python src/final_analysis.py   --base $B     # 종합 표
```

### 검정 결과가 논문과 같은지 확인

순열 검정은 조건당 6표본일 때 $\binom{12}{6}=924$가지를 전수 열거하므로
난수에 의존하지 않는다. 다음 값이 그대로 나와야 한다.

| 설정 | $\Delta_{KO}$ | $X$ |
|---|---|---|
| 7.8B GPTQ (18×18) | +0.50, *p*=0.171 | +1.16, *p*=0.016 |
| 7.8B AWQ (6×6 전수) | +1.04, *p*=0.054 | +1.65, *p*=0.024 |
| 32B GPTQ (6×6 전수) | +1.03, *p*=0.113 | +1.23, *p*=0.043 |

부트스트랩은 난수를 쓰므로 소수 셋째 자리까지 일치하지는 않는다. 시드는 각
스크립트의 `--seed` 기본값이다.

---

## 2층. 평가만 재현

기존 체크포인트를 다시 평가한다. 문항별 예측 기록을 반드시 남긴다
(`--log_samples`) — 대응 부트스트랩이 이것을 요구한다.

```bash
source venv-quant/bin/activate
CUDA_VISIBLE_DEVICES=0 bash scripts/seed_job.sh eval:gptq:ko:1
CUDA_VISIBLE_DEVICES=0 bash scripts/seed_job.sh evalone:gptq_4bit_s1:gptq_ko_s1:ko
```

한국어 평가에는 `HF_HUB_OFFLINE=1`이 자동으로 걸린다. KMMLU 원격 검증 과정에서
일시적 404가 발생한 적이 있고 로컬 캐시는 정상이었기 때문이다.

---

## 1층. 전체 재현

### 0) 사전 점검

```bash
bash scripts/preflight_32b.sh quick     # 설정·모듈 구조·토크나이저, GPU 불필요
bash scripts/preflight_32b.sh full      # 실제 GPTQ 양자화 8표본까지
```

`quick`이 실패하면 진행하지 않는다. 대개 원격 코드 리비전 문제이며,
`src/find_revision.py`로 호환 리비전을 찾는다.

### 1) BF16 기준선

저하율의 분모이므로 가장 먼저 확보한다.

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/seed_job.sh bf16:ko
CUDA_VISIBLE_DEVICES=1 bash scripts/seed_job.sh bf16:en
```

### 2) 7.8B 시드 반복 (18시드 × 2언어, GPTQ)

```bash
mkdir -p logs
for g in 0 1 2 3; do
  nohup bash scripts/queue_worker.sh $g scripts/jobs/jobs_78b.txt \
    > logs/gpu$g.txt 2>&1 & sleep 5
done
```

동적 큐다. 각 워커가 `mkdir`로 작업을 원자적으로 선점하므로, 지금 빈 GPU로
먼저 시작하고 다른 GPU가 비면 같은 명령으로 합류시켜도 된다. 선점 기록은
`queue_claims/<작업파일명>/`에 남고, 같은 작업을 다시 돌리려면 해당 디렉터리를
지운다.

### 3) 32B 규모 확장 (6시드 × 2언어)

```bash
MODEL_ID=LGAI-EXAONE/EXAONE-3.5-32B-Instruct \
REVISION=d6fa88cd8d2c9512b40578bdc44e64909e5a5042 \
BASE=quant_experiments/EXAONE-3.5-32B-Instruct \
GPU_UTIL=0.90 \
nohup bash scripts/queue_worker.sh 0 scripts/jobs/jobs_32b.txt \
  > logs/32b_gpu0.txt 2>&1 &
```

`GPU_UTIL`은 32B에서 0.90이다(BF16 가중치만 약 64 GB). 7.8B와 비교하려면
7.8B 쪽은 0.70을 유지해야 한다 — 여기서 바꾸면 기존 결과와 조건이 어긋난다.

작업당 대략적 소요: 양자화 1시간 15분 + KMMLU 2시간 40분 + MMLU 1시간 50분
≈ 5시간 45분. 12작업을 4장에 나누면 약 17시간이다. KMMLU는 35,030문항으로
MMLU(14,042)의 2.5배라 가장 오래 걸린다.

### 4) 캘리브레이션 크기 실험

```bash
nohup bash scripts/queue_worker.sh 0 scripts/jobs/e2_jobs.txt > logs/e2.txt 2>&1 &
```

### 5) 산출물 검증

```bash
bash scripts/verify_run.sh quant_experiments/EXAONE-3.5-7.8B-Instruct
```

토크나이저 일관성, 표본 집합 해시의 상이성, 평가 기록의 문항 수를 확인한다.

### 6) 배포용 데이터 추출

```bash
bash scripts/make_release.sh /path/to/quant_exp .
python src/hash_manifest.py --root . --out MANIFEST.sha256
```

---

## 진행 상황 확인

워커 로그의 진행바는 `\r`로 갱신되므로 `tail -n`으로는 한 줄로 뭉쳐 보인다.

```bash
for g in 0 1 2 3; do
  echo "== GPU $g =="
  tail -c 600 logs/gpu$g.txt | tr '\r' '\n' | tail -3
done
```

GPU별로 어떤 프로세스가 붙어 있는지는 `ps aux`로는 알 수 없다
(`CUDA_VISIBLE_DEVICES`는 환경변수라 커맨드라인에 나타나지 않는다).

```bash
for g in 0 1 2 3; do
  echo -n "GPU $g: "
  nvidia-smi -i $g --query-compute-apps=pid,used_memory --format=csv,noheader | tr '\n' ' '
  echo
done
```

남은 작업 수:

```bash
echo "전체 $(grep -vc '^#\|^$' scripts/jobs/jobs_32b.txt) / 선점 $(ls queue_claims/jobs_32b 2>/dev/null | wc -l)"
```

---

## 알려진 함정

| 증상 | 원인과 대응 |
|---|---|
| `ImportError: RopeParameters` | 상류 원격 코드가 transformers v5 방식으로 갱신되었다. 리비전을 고정한다 |
| 음의 저하율(4비트가 BF16보다 정확) | 동률 첫 선택 채점의 인공물이다. `rescore_all.py`로 분할 채점을 쓴다 |
| `int('(A)')` 오류 | HAE-RAE는 정답이 문자열이다. `target_index()`가 처리한다 |
| KMMLU 로딩 404 | 로컬 캐시는 정상이다. `HF_HUB_OFFLINE=1`로 우회한다 |
| 저장된 원격 코드가 최신판으로 덮어써짐 | `save_pretrained()`의 동작이다. `_pin_remote_code_files()`가 저장 직후 고정 리비전으로 되돌린다 |
| 32B 양자화 중 OOM | `GPU_UTIL`을 낮추지 말고(비교 조건이 어긋난다) 동시 실행 수를 줄인다 |
| 문항 순서 sha256 불일치 | 실행 간 평가 문항 집합이 다르다. 대응 부트스트랩의 전제가 깨지므로 해당 실행을 다시 평가한다 |
