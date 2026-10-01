# 실험 중간 수정 이력

실험이 7.8B에서 32B로 확장될 때, 7.8B를 전제로 하드코딩된 부분을 풀어야 했다.
수정 사실을 감추지 않기 위해 패치와 적용 스크립트를 그대로 남긴다.
`src/` 아래 파일들은 **수정이 적용된 최종 상태**다.

| 대상 | 무엇을 고쳤는가 | 왜 중요한가 |
|---|---|---|
| `activation_analysis.py` | `MODEL_ID`/`REVISION`을 환경변수로 덮어쓸 수 있게 | `ppl_eval.py`가 이 두 값을 import해 BF16 기준 모델을 정한다. 하드코딩 상태로 32B를 재면 양자화는 32B인데 BF16만 7.8B가 되어 퍼플렉시티 증가율이 무의미해진다 |
| `rescore_all.py` | 정합성 점검 참조값을 그 실행의 `results*.json`에서 읽도록 | 7.8B 보고치가 하드코딩되어 있어 32B 재채점에서 항상 불일치로 표시되었다 |
| `make_figure1.py` | 하드코딩된 7.8B X 값 대신 CLI 인자 | 그림을 다른 규모로 다시 그릴 수 없었다 |

`apply_fixes.py`는 멱등하다. 이미 적용된 파일은 건너뛰고, 적용 전 `.bak`을 남기며
적용 후 문법 검사를 한다.

```bash
python docs/patches/apply_fixes.py            # 무엇을 바꿀지만 표시
python docs/patches/apply_fixes.py --write    # 적용
python docs/patches/apply_fixes.py --revert   # .bak 으로 복원
```

퍼플렉시티 측정 코드 변경이 결과에 미친 영향은 논문 부록 B.8에 별도로 보고했다.
