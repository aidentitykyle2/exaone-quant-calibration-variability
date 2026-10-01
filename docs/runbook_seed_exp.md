# ①②③ 실험 실행 절차서 (EXAONE 3.5-7.8B, H100 4장)

목표: 시드 반복(①), 기제 분석(②), 대응 부트스트랩(③)을 한 번의 배치로 확보해
"캘리브레이션 언어 효과가 표본 추출의 우연이 아니며, 활성화 수준의 기제로
설명된다"는 주장의 근거를 갖춘다.

---

## 전체 일정

| 구간 | 내용 | 소요 |
|---|---|---|
| 0단계 | 파일 배치 + 사전 점검 3종 | 약 20분 |
| 1단계 | 4-GPU 배치 실행 | **약 8시간 10분** |
| 　├ 초반 2.7시간 | ② 기제 분석, BF16·seed 0 재평가 전부 완료 | → 중간 점검 가능 |
| 　└ 이후 5.5시간 | ① seed 1·2 양자화 + 평가 | |
| 2단계 | 부트스트랩 + 결과 확인 | 약 5분 |
| **합계** | | **약 8시간 30분** |

### GPU별 작업 큐

| GPU | 작업 순서 | 소요 |
|---|---|---|
| 0 | 분석(60) → seed0 GPTQ[ko] 재평가(95) → AWQ[ko] s1(200) → GPTQ[ko] s1(135) | 8.2h |
| 1 | BF16 한국어(55) → seed0 GPTQ[en_i] 재평가(95) → AWQ[ko] s2(200) → GPTQ[ko] s2(135) | 8.1h |
| 2 | BF16 영어(40) → seed0 AWQ[ko] 재평가(95) → AWQ[en_i] s1(200) → GPTQ[en_i] s1(135) | 7.8h |
| 3 | seed0 AWQ[en_i] 재평가(95) → AWQ[en_i] s2(200) → GPTQ[en_i] s2(135) | 7.2h |

짧은 작업을 앞에 둔 이유: **약 2.7시간 뒤 ②와 ③(seed 0)이 이미 가능한 상태**가 된다. 긴 시드 반복을 기다리지 않고 중간 결과로 방향을 점검할 수 있다.

---

## 파일

| 파일 | 역할 | 상태 |
|---|---|---|
| `run_quant_stage.py` | `--calib-seed` 추가 | **수정** |
| `seed_job.sh` | 작업 하나 실행 (bf16/eval/lane/analysis) | 신규 |
| `launch_seed_exp.sh` | 4-GPU 큐 분배 | 신규 |
| `activation_analysis.py` | ② M1·M2·M3 | 신규 |
| `paired_bootstrap.py` | ③ 문항 부트스트랩 + 2단계 부트스트랩 | 신규 |

### `run_quant_stage.py` 변경 요지

- `--calib-seed 0`: 데이터셋 앞에서부터 결정적 추출. **시드 도입 전 코드와 샘플이 완전히 동일함을 검증했다.** 따라서 기존 결과를 seed 0으로 그대로 재사용한다.
- `--calib-seed 1, 2`: 셔플 후 추출. 산출물에 `_s1`, `_s2` 접미사가 붙어 기존 결과를 덮어쓰지 않는다.
- C4처럼 스트리밍 데이터셋은 버퍼 셔플(10,000)을 쓴다.

---

## 0단계. 사전 점검 (약 20분, 반드시 순서대로)

```bash
cd /path/to/quant_exp
source venv-quant/bin/activate
export HF_HOME=/path/to/.cache/huggingface
echo "venv: $VIRTUAL_ENV"      # /path/to/quant_exp/venv-quant 확인
```

### 점검 1 — 파일 배치 확인

```bash
grep -c "calib_seed" run_quant_stage.py        # 10 이상이어야 한다
ls seed_job.sh launch_seed_exp.sh activation_analysis.py paired_bootstrap.py
chmod +x seed_job.sh launch_seed_exp.sh
```

### 점검 2 — 시드별 캘리브레이션 로딩 (GPU 미사용, 2분)

```bash
CUDA_VISIBLE_DEVICES="" HF_HUB_OFFLINE=1 python -c "
from run_quant_stage import load_calib_dataset
for lang in ['ko','en_instruct']:
    a = load_calib_dataset(lang, 8, seed=0)['text']
    b = load_calib_dataset(lang, 8, seed=1)['text']
    print(lang, '| seed0≠seed1:', a != b, '| 예시:', b[0][:60])
"
```

`seed0≠seed1: True`가 두 언어 모두 나와야 한다. 오프라인 로딩 오류가 나면 `HF_HUB_OFFLINE=1`을 빼고 한 번 실행해 캐시를 채운다.

### 점검 3 — 기제 분석 스모크 테스트 (GPU 1장, 약 10분)

```bash
CUDA_VISIBLE_DEVICES=0 python activation_analysis.py \
  --base quant_experiments/EXAONE-3.5-7.8B-Instruct \
  --out /tmp/mech_smoke --smoke
```

**이 점검이 가장 중요하다.** M3은 compressed-tensors 체크포인트를 HuggingFace transformers로 불러오는데, 이 경로는 아직 실제 환경에서 검증되지 않았다. 판정은 다음과 같다.

| 결과 | 조치 |
|---|---|
| M1·M2·M3 요약이 모두 출력 | 정상 — 1단계로 |
| M3에서 `로딩 실패` | 분석 작업을 큐에서 빼고 배치 실행, 분석은 별도로 디버깅 |
| M1·M2부터 실패 | 모듈 이름 규칙 문제 — 오류 메시지 공유 |

M3이 실패할 경우의 큐 조정:

```bash
QUEUE_0="eval:gptq:ko:0 lane:awq:ko:1 lane:gptq:ko:1" bash launch_seed_exp.sh
```

---

## 1단계. 배치 실행 (약 8시간 10분)

```bash
nvidia-smi                      # 4장 모두 비었는지 확인
bash launch_seed_exp.sh         # GPU 상태를 보여주고 확인을 받는다
```

GPU를 3분 간격으로 시작한다(동시에 모델 가중치를 읽으며 디스크가 경합하는 것을 피하기 위함).

### 진행 확인

```bash
grep -h '########' logs_seed/gpu*.txt     # 작업 시작·종료
nvidia-smi
tail -3 logs_seed/gpu0.txt
```

모든 작업이 `exit=0`으로 끝나야 한다. 작업 하나가 실패해도 같은 GPU의 다음 작업은 계속 진행된다.

---

## 2단계-A. 중간 점검 (시작 후 약 2.7시간)

이 시점에 BF16과 seed 0 재평가 4건, 기제 분석이 끝나 있다.

```bash
# ② 기제 분석 결과
cat quant_experiments/EXAONE-3.5-7.8B-Instruct/mechanism/summary.txt

# ③ seed 0만으로 부트스트랩 (시드 반복 전 결과)
python paired_bootstrap.py --base quant_experiments/EXAONE-3.5-7.8B-Instruct
```

**정합성 점검을 반드시 볼 것.** 부트스트랩 출력 첫 부분에 seed 0 재평가의 정확도가 나온다. 기존 보고치(KMMLU: BF16 0.4555, GPTQ[ko] 0.4484, AWQ[ko] 0.4200)와 일치해야 한다. 소수점 넷째 자리 수준의 차이는 vLLM 연산 비결정성으로 허용되지만, 그 이상 다르면 파이프라인에 문제가 있다는 뜻이므로 배치를 계속 돌리기 전에 원인을 확인해야 한다.

## 2단계-B. 최종 (배치 완료 후)

```bash
python paired_bootstrap.py --base quant_experiments/EXAONE-3.5-7.8B-Instruct
```

시드 0·1·2가 모두 있으면 2단계 부트스트랩까지 계산된다.

---

## 결과 해석 기준 (결과를 보기 전에 고정)

### ① + ③ 부트스트랩

| 항목 | 판정 기준 |
|---|---|
| 2단계 부트스트랩 L_KO 구간 | 0을 포함하지 않으면 한국어 언어 효과 확정 |
| 2단계 부트스트랩 X 구간 | 0을 포함하지 않으면 **교차 확정** — 논문의 핵심 주장 |
| 시드간 SD / 문항 SE 비율 | 1 이상이면 캘리브레이션 표본 변동이 평가 표집만큼 크다 → 시드 반복이 필수였음을 입증 |

H1 판정 상향 조건: **2단계 부트스트랩에서 X의 95% 구간이 0을 포함하지 않을 것.** 이것이 충족되면 "방향성 지지"를 "지지"로 올릴 수 있다.

### ② 기제 분석

| 측정 | 기제를 지지하는 결과 |
|---|---|
| M1 Jaccard | 언어 간 < 언어 내 기준선 (단, 기준선 자체가 낮게 나올 수 있어 보조 지표) |
| **M2 Spearman** | **언어 간 < 언어 내 기준선 — 주 지표** |
| **M3 레이어 오차** | **Δ_KO > 0 이고 Δ_EN > 0 — 정확도 교차와 같은 방향의 교차가 은닉 상태에서 존재** |

M2를 주 지표로 두는 이유: 모의 검증에서 상위 1% 채널 집합(M1)은 같은 언어 안에서도 Jaccard 0.22 수준으로 크게 흔들린 반면, 순위 상관(M2)은 0.93으로 안정적이었다.

M3이 가장 중요한 이유: M1·M2는 "분포가 다르다"까지만 보여준다. M3은 그 차이가 실제로 **언어별 양자화 오차로 이어지는지**를 보여주어 기제와 정확도 결과를 잇는다.

### 기제 분석이 지지하지 않는 결과가 나온다면

그것도 보고해야 한다. 예를 들어 M2 언어 간 상관이 기준선과 차이가 없다면 "캘리브레이션 언어 효과는 헤시안 대각 통계로는 설명되지 않는다"가 결론이 되며, 비대각 성분이나 다른 기제를 향후 과제로 남긴다. 부정적 결과도 선행연구(Chimoto 등)의 기제 가설을 검증한 것이므로 기여가 된다.

---

## 논문 반영 지점

| 절 | 반영 내용 |
|---|---|
| 3.3 | 시드 반복 설계(seed 0·1·2) 서술 |
| 3.5 | 부트스트랩 방법(문항 대응 재표집, 2단계 재표집) 서술 |
| 5.2 | 이항 근사 σ를 **부트스트랩 구간으로 교체** |
| 5.x 신설 | 시드 간 변동 표 |
| 6.2 | "기제 가설"을 **M1·M2·M3 측정 결과로 교체** |
| 6.1 | H1 판정 상향 검토 |
| 6.5 | 한계 "문항 단위 분석 부재", "단일 시행", "기제 미검증" 세 항목 해소 |

## 안전 수칙

- 다운로드 중 Ctrl+C 금지 (캐시 손상 → 원인 불명 404 전례)
- 정리할 때는 `pkill -f lm_eval` 과 `pkill -f EngineCore` 를 함께 (전자만으로는 vLLM 엔진이 남는다)
- 7.8B 평가는 `gpu_memory_utilization=0.70` 유지. 0.90은 32B 전용이며, 여기서 바꾸면 기존 결과와 조건이 어긋난다
