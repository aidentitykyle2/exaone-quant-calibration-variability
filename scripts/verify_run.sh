#!/bin/bash
# verify_run.sh — 시드 반복 실험 산출물의 완전성 검증
#
# 분석에 들어가기 전에 "exit 0이지만 내용이 잘린" 산출물을 걸러낸다.
# 재현성을 주제로 하는 논문은 자기 산출물의 완전성을 먼저 보여야 하므로,
# 이 스크립트의 출력을 그대로 부록에 실을 수 있게 구성했다.
#
# 점검 항목
#   V1 작업 종료코드      선점 디렉터리의 exit_code가 모두 0인지
#   V2 체크포인트         조건별 디렉터리와 safetensors 존재
#   V3 평가 산출물        results json + samples jsonl 존재, 줄 수가 0이 아닌지
#                         (samples jsonl이 tie_split 재채점의 유일한 입력이다)
#   V4 캘리브레이션 매니페스트
#                         조건별 존재, 표본 수 일치, **set_sha256이 전부 상이**
#                         — 두 시드가 같은 표본을 뽑았다면 SD가 과소추정된다.
#                           단일 실행 변동을 논하는 논문에서 치명적인 오류이므로
#                           여기서 반드시 잡는다.
#   V5 리비전 고정        results.json의 environment에 model_revision이 남아 있는지
#
# 사용법
#   BASE=quant_experiments/EXAONE-3.5-32B-Instruct SEEDS="1 2 3 4 5 6" \
#     bash verify_run.sh
#   BASE=quant_experiments/EXAONE-3.5-7.8B-Instruct SEEDS="$(seq 1 18)" \
#     CLAIMS=queue_claims/jobs_78b bash verify_run.sh

set -u
BASE="${BASE:-quant_experiments/EXAONE-3.5-32B-Instruct}"
SEEDS="${SEEDS:-1 2 3 4 5 6}"
LANGS="${LANGS:-ko en_instruct}"
CALIB_N="${CALIB_N:-256}"
CLAIMS="${CLAIMS:-queue_claims/jobs_32b}"
REPORT="${REPORT:-${BASE}/verify_report.txt}"

fail=0
note () { echo "$@"; echo "$@" >> "${REPORT}.tmp"; }

: > "${REPORT}.tmp"
note "산출물 검증: ${BASE}"
note "시드: ${SEEDS} | 언어: ${LANGS} | 표본 수: ${CALIB_N}"
note "시각: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
note ""

# ---------- V1 작업 종료코드 ----------
note "=== V1. 작업 종료코드 ==="
if [ -d "${CLAIMS}" ]; then
  n_claim=$(find "${CLAIMS}" -mindepth 1 -maxdepth 1 -type d | wc -l)
  n_code=$(find "${CLAIMS}" -name exit_code | wc -l)
  n_zero=$(grep -rlx 0 "${CLAIMS}"/*/exit_code 2>/dev/null | wc -l)
  note "  선점 ${n_claim} | 종료코드 기록 ${n_code} | 정상(0) ${n_zero}"
  if [ "${n_code}" != "${n_zero}" ]; then
    note "  [실패] 0이 아닌 종료코드가 있다:"
    for f in "${CLAIMS}"/*/exit_code; do
      [ -f "${f}" ] || continue
      c=$(cat "${f}")
      [ "${c}" = "0" ] || note "    $(basename "$(dirname "${f}")") -> ${c}"
    done
    fail=1
  elif [ "${n_claim}" != "${n_code}" ]; then
    note "  [주의] 종료코드가 없는 선점이 있다(실행 중이거나 워커가 죽었다)."
    fail=1
  else
    note "  [통과]"
  fi
else
  note "  [건너뜀] 선점 디렉터리 없음 → ${CLAIMS}"
fi
note ""

# ---------- V2 체크포인트 ----------
note "=== V2. 체크포인트 ==="
miss=0
for s in ${SEEDS}; do
  for l in ${LANGS}; do
    d="${BASE}/gptq_4bit"
    [ "${l}" != "ko" ] && d="${d}_calib_${l}"
    [ "${CALIB_N}" != "256" ] && d="${d}_n${CALIB_N}"
    d="${d}_s${s}"
    if [ ! -d "${d}" ]; then
      note "  [없음] ${d}"; miss=$((miss+1))
    elif [ -z "$(find "${d}" -name '*.safetensors' -print -quit)" ]; then
      note "  [safetensors 없음] ${d}"; miss=$((miss+1))
    fi
  done
done
if [ ${miss} -eq 0 ]; then note "  [통과] 전 조건 존재"; else fail=1; fi
note ""

# ---------- V3~V5 ----------
note "=== V3~V5. 평가 산출물 · 매니페스트 · 리비전 ==="
python - "${BASE}" "${CALIB_N}" "${SEEDS}" "${LANGS}" <<'PY' | tee -a "${REPORT}.tmp"
import hashlib, json, sys
from collections import defaultdict
from pathlib import Path

base = Path(sys.argv[1]); calib_n = int(sys.argv[2])
seeds = sys.argv[3].split(); langs = sys.argv[4].split()
bad = 0

# ---- V3 평가 산출물
print("--- V3. 평가 산출물 (results json + samples jsonl)")
names = ["bf16"] + [f"gptq_{l}_s{s}" for s in seeds for l in langs]
rows = []
for evlang in ("ko", "en"):
    for name in names:
        d = base / "seed_eval" / evlang / name
        if not d.is_dir():
            print(f"  [없음] {d}"); bad = 1; continue
        res = sorted(d.rglob("results*.json"))
        smp = sorted(d.rglob("samples*.jsonl"))
        if not res:
            print(f"  [results 없음] {d}"); bad = 1; continue
        if not smp:
            print(f"  [samples 없음] {d}  <- tie_split 재채점 불가"); bad = 1; continue
        lines = sum(1 for f in smp for _ in f.open(encoding="utf-8", errors="ignore"))
        if lines == 0:
            print(f"  [samples 비어 있음] {d}"); bad = 1; continue
        rows.append((evlang, name, len(smp), lines))
if rows:
    per = defaultdict(list)
    for evlang, name, nf, nl in rows:
        per[evlang].append(nl)
    for evlang, v in per.items():
        lo, hi = min(v), max(v)
        flag = "" if lo == hi else "  <- 조건 간 문항 수가 다르다. 대응 부트스트랩에서 교집합만 쓰이는지 확인할 것"
        print(f"  {evlang}: {len(v)}개 조건, 문항 수 {lo}~{hi}{flag}")
        if lo != hi:
            bad = 1
print(f"  확인한 조건 {len(rows)} / 기대 {2*len(names)}")

# ---- V4 캘리브레이션 매니페스트
print("\n--- V4. 캘리브레이션 매니페스트 (표본 중복 여부)")
mdir = base / "calib_manifests"
shas, counts = {}, {}
for s in seeds:
    for l in langs:
        stem = f"calib_{l}_n{calib_n}_s{s}"
        jl, mj = mdir / f"{stem}.jsonl", mdir / f"{stem}.meta.json"
        if not jl.exists():
            print(f"  [없음] {jl}"); bad = 1; continue
        recs = [json.loads(x) for x in jl.read_text(encoding="utf-8").splitlines() if x.strip()]
        counts[stem] = len(recs)
        if len(recs) != calib_n:
            print(f"  [표본 수 불일치] {stem}: {len(recs)} != {calib_n}"); bad = 1
        if mj.exists():
            meta = json.loads(mj.read_text(encoding="utf-8"))
            sha = meta.get("set_sha256")
        else:
            sha = None
            print(f"  [meta 없음] {mj}  <- 부록에 실을 표본 지문이 없다"); bad = 1
        if sha is None:  # meta가 없으면 본문에서 직접 계산해 대체한다
            h = hashlib.sha256()
            for r in recs:
                h.update(str(r.get("source_id")).encode())
                h.update(b"\x00")
            sha = h.hexdigest()
        shas[stem] = sha

dup = defaultdict(list)
for k, v in shas.items():
    dup[v].append(k)
collide = {v: ks for v, ks in dup.items() if len(ks) > 1}
if collide:
    print("  [실패] 서로 다른 조건이 같은 표본 집합을 쓰고 있다:")
    for v, ks in collide.items():
        print(f"    {v[:16]}... : {ks}")
    print("    시드가 실제로 적용되지 않았다는 뜻이며, 이 상태의 SD는 과소추정이다.")
    bad = 1
elif shas:
    print(f"  [통과] {len(shas)}개 조건의 표본 집합이 모두 상이하다")
    for k in sorted(shas):
        print(f"    {k}: n={counts[k]} sha={shas[k][:16]}...")

# ---- V5 리비전 고정
print("\n--- V5. 리비전 고정 기록")
rf = base / "results.json"
if not rf.exists():
    print(f"  [없음] {rf}"); bad = 1
else:
    env = json.loads(rf.read_text(encoding="utf-8")).get("environment", {})
    rev = env.get("model_revision")
    print(f"  model_revision: {rev}")
    if not rev or rev == "main":
        print("  [주의] 리비전이 고정되어 있지 않다. 재현 부록에 커밋 해시를 남길 수 없다.")
        bad = 1
    for k in ("calib_lang", "calib_seed", "calib_n", "calib_dataset"):
        if k in env:
            print(f"  {k}: {env[k]}")
    print("  (results.json의 environment는 마지막 실행 조건만 담는다. 조건별 기록은")
    print("   efficiency 항목과 calib_manifests에 남아 있다.)")

sys.exit(1 if bad else 0)
PY
rc=$?
[ ${rc} -ne 0 ] && fail=1
note ""

mkdir -p "$(dirname "${REPORT}")"
mv "${REPORT}.tmp" "${REPORT}"
if [ ${fail} -eq 0 ]; then
  note "=== 전체 통과. 재채점 단계로 진행해도 된다. ==="
  echo "보고서: ${REPORT}"
  exit 0
fi
echo "=== 실패 항목이 있다. 위 내용을 해결한 뒤 분석을 시작할 것. ==="
echo "보고서: ${REPORT}"
exit 1
