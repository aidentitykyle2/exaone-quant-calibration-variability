# 초기 단계 스크립트

**논문 결과를 만든 스크립트가 아니다.** 연구 초기의 기법 비교(AWQ/GPTQ/QuaRot
단일 실행) 단계에서 사용했고, 주제가 재현성 분석으로 옮겨간 뒤 `seed_job.sh` +
`queue_worker.sh` 조합으로 대체되었다.

이력 보존과 효율성 측정값(논문 부록 B.7)의 출처 확인을 위해 남긴다. 재현을
시도할 때는 이 디렉터리가 아니라 상위 `scripts/`를 쓸 것.

| 파일 | 당시 역할 |
|---|---|
| `run_all_stages.sh` | BF16 → AWQ → GPTQ → QuaRot 순차 실행 |
| `run_calib_ablation.sh` | 캘리브레이션 언어 ablation 초기판 |
| `run_eval_english.sh` | 영어 벤치마크 평가 추가 |
| `run_lane.sh`, `launch_lanes.sh` | GPU별 고정 레인 배분 (동적 큐 이전 방식) |
| `launch_seed_exp.sh` | 시드 반복 실험 초기 런처 |
| `run_quarot_en.sh` | QuaRot 영어 캘리브레이션 조건 |
