#!/usr/bin/env python3
"""
make_figure1.py — 논문 그림 1: 캘리브레이션 표본별 언어 비대칭

시드별 X를 점으로, 두 종류의 95% 구간을 나란히 표시한다.
  단일 실행  : 표본 하나로 평가 문항만 재표집 (좁고 0에서 멀다)
  표본 반영  : 문항과 표본을 함께 재표집 (0에 가깝다)
이 대비가 논문의 핵심 메시지다.
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

# 7.8B 기본값. 다른 모델(32B 등)은 --x / --single / --multi 또는 --json으로 넘긴다.
#   single_run_risk.py -> SINGLE (단일 실행, 문항만 재표집)
#   final_analysis.py  -> MULTI  (두 단계: 문항+표본)
X_DEFAULT = [3.29,-2.45,0.71,2.15,3.03,-0.58,0.41,-0.44,1.20,2.89,
             2.46,0.01,0.06,-1.24,1.14,-0.41,5.74,2.94]
SINGLE_DEFAULT = (3.27, 2.20, 4.33)
MULTI_DEFAULT  = (1.20, 0.12, 2.16)

_ap = argparse.ArgumentParser(description="논문 그림: 캘리브레이션 표본별 언어 비대칭")
_ap.add_argument("--x", nargs="+", type=float, help="시드별 X 값")
_ap.add_argument("--single", nargs=3, type=float, metavar=("M", "LO", "HI"),
                 help="단일 실행 구간 (평균 하한 상한)")
_ap.add_argument("--multi", nargs=3, type=float, metavar=("M", "LO", "HI"),
                 help="표본 반영 구간 (평균 하한 상한)")
_ap.add_argument("--json", help='{"x": [...], "single": [m,lo,hi], "multi": [m,lo,hi]} 파일')
_ap.add_argument("--out", default="figure1_asymmetry", help="출력 파일 이름(확장자 제외)")
_ap.add_argument("--label", default=None, help="제목에 덧붙일 모델 이름(예: EXAONE 3.5-32B)")
_a = _ap.parse_args()

if _a.json:
    _d = json.loads(open(_a.json, encoding="utf-8").read())
    X = list(map(float, _d["x"]))
    SINGLE = tuple(map(float, _d["single"]))
    MULTI = tuple(map(float, _d["multi"]))
else:
    X = _a.x if _a.x else X_DEFAULT
    SINGLE = tuple(_a.single) if _a.single else SINGLE_DEFAULT
    MULTI = tuple(_a.multi) if _a.multi else MULTI_DEFAULT
if _a.x and not (_a.single and _a.multi):
    raise SystemExit("--x를 주면 --single과 --multi도 함께 주어야 한다. "
                     "구간은 single_run_risk.py와 final_analysis.py 출력에서 가져올 것.")

# 한글 폰트가 없으면 영문 라벨로 대체
# 이름만으로 판정하면 대체 폰트로 렌더링되어 글자가 깨진다.
# 실제로 해당 폰트 파일이 해결되는지 확인한다.
ko = False
for cand in ("NanumGothic", "NanumBarunGothic", "Noto Sans CJK KR",
             "Noto Sans KR", "Malgun Gothic", "AppleGothic"):
    try:
        f = font_manager.findfont(font_manager.FontProperties(family=cand),
                                  fallback_to_default=False)
        if f:
            plt.rcParams["font.family"] = cand
            plt.rcParams["axes.unicode_minus"] = False
            ko = True
            break
    except Exception:
        continue
    L = dict(xl="캘리브레이션 표본(시드)", yl="언어 비대칭 X (%p)",
             single="단일 실행\n(문항만 재표집)", multi=f"{len(X)}표본\n(문항+표본)",
             neg="방향이 뒤바뀐 표본", ttl="캘리브레이션 표본에 따른 언어 비대칭의 변동")
else:
    L = dict(xl="Calibration sample (seed)", yl="Language asymmetry X (pp)",
             single="Single run\n(items only)", multi=f"{len(X)} samples\n(items+samples)",
             neg="Sign-reversed samples", ttl="Variation of language asymmetry across calibration samples")

fig, (ax, bx) = plt.subplots(1, 2, figsize=(11, 4.2),
                             gridspec_kw={"width_ratios": [3, 1]}, sharey=True)

ax.axhline(0, color="0.3", lw=1)
neg = [i for i, v in enumerate(X) if v < 0]
ax.scatter(range(len(X)), X, s=70, color="#2b6cb0", zorder=3, label=None)
ax.scatter(neg, [X[i] for i in neg], s=70, facecolors="none",
           edgecolors="#c53030", linewidths=2, zorder=4, label=L["neg"])
ax.axhline(np.mean(X), color="#2b6cb0", ls="--", lw=1, alpha=.7)
ax.set_xticks(range(len(X)))
_ttl = L["ttl"] + (f"  [{_a.label}]" if _a.label else "")
ax.set_xlabel(L["xl"]); ax.set_ylabel(L["yl"]); ax.set_title(_ttl, fontsize=11)
ax.legend(loc="upper left", fontsize=9, frameon=False)
ax.grid(axis="y", alpha=.25)

for i, (m, lo, hi, col, lab) in enumerate([
        (*SINGLE, "#b7791f", L["single"]), (*MULTI, "#2b6cb0", L["multi"])]):
    bx.errorbar(i, m, yerr=[[m-lo],[hi-m]], fmt="o", color=col,
                ms=9, capsize=7, lw=2.2)
    bx.text(i, hi+0.35, f"{m:+.2f}", ha="center", fontsize=10, color=col)
bx.axhline(0, color="0.3", lw=1)
bx.set_xlim(-.6, 1.6); bx.set_xticks([0, 1])
bx.set_xticklabels([L["single"], L["multi"]], fontsize=9)
bx.grid(axis="y", alpha=.25)

fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(f"{_a.out}.{ext}", dpi=200, bbox_inches="tight")
print(f"저장: {_a.out}.png / .pdf  (한글 폰트 {'사용' if ko else '없음 → 영문 라벨'})")
print(f"  표본 {len(X)}개, 음수 {len(neg)}개, 평균 {np.mean(X):+.2f}, 범위 [{min(X):+.2f}, {max(X):+.2f}]")
