#!/usr/bin/env python3
"""
make_figure_risk.py — 논문 그림: 귀무 짝과 교차 짝의 효과 크기 분포

single_run_risk.py가 저장한 CSV를 읽어, 참 효과가 0인 귀무 짝에서 나온 값들이
실제 교차 짝과 같은 크기에 도달하는지를 보여 준다. 이 그림 하나가 논문의
핵심 주장을 담는다: 단일 실행으로 얻은 값은 실제 효과와 우연을 구분하지 못한다.

읽는 파일
  <base>/risk/single_run_risk_<method>_<scoring>_<seeds>.csv
  열: n,pair_type,cond_a,cond_b,quantity,point,ci_low,ci_high,significant

설계 근거
  - 형태: n이 30~36개라 상자그림·바이올린은 과도한 평활을 일으킨다. 개별 점을
          흩뿌리고(스트립) 중앙값만 선으로 표시한다.
  - 인코딩: 짝 종류는 x 위치와 축 라벨로, 유의 여부는 채움/비움으로 나눈다.
            색은 보조 수단이므로 색만으로 구분해야 하는 정보가 없다.
  - 겹침: 점마다 흰 테두리를 둘러 밀집 구간에서도 개수를 셀 수 있게 한다.
  - 흔들림(jitter)은 고정 시드로 생성해 같은 그림이 재현되게 한다.

사용법
  python make_figure_risk.py --csv quant_experiments/EXAONE-3.5-32B-Instruct/risk/\\
single_run_risk_gptq_tie_split_all.csv --out figure3_risk_32b --label "EXAONE 3.5-32B"
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.lines import Line2D

# 기존 그림 1과 같은 두 색. 검증 결과 protan ΔE 19.7 / 정상 28.8로 기준을 넘긴다.
C_NULL = "#c53030"    # 귀무 짝 — 참 효과가 0인데 유의하게 나온 것이 문제의 핵심
C_CROSS = "#2b6cb0"   # 교차 짝
INK = "#1a202c"
INK_MUTED = "#4a5568"
SURFACE = "#ffffff"

QUANTS = ["L_KO", "L_EN", "X"]


def pick_font() -> bool:
    for cand in ("NanumGothic", "NanumBarunGothic", "Noto Sans CJK KR",
                 "Noto Sans KR", "Malgun Gothic", "AppleGothic"):
        try:
            if font_manager.findfont(font_manager.FontProperties(family=cand),
                                     fallback_to_default=False):
                plt.rcParams["font.family"] = cand
                plt.rcParams["axes.unicode_minus"] = False
                return True
        except Exception:
            continue
    return False


def load(path: str):
    """{(quantity, pair_type): [(point, significant), ...]} 와 n 집합을 돌려준다."""
    rows = defaultdict(list)
    ns = set()
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            try:
                pt = float(r["point"])
                sig = int(float(r["significant"]))
            except (TypeError, ValueError, KeyError):
                continue
            ns.add(r.get("n", ""))
            rows[(r["quantity"], r["pair_type"])].append((pt, sig))
    return rows, ns


def strip(ax, x0, vals, color, rng, label_n=True):
    """한 그룹의 점을 흩뿌린다. 유의=채움, 비유의=비움."""
    if not vals:
        return 0.0
    pts = np.array([v for v, _ in vals], dtype=float)
    sig = np.array([s for _, s in vals], dtype=bool)
    jit = rng.uniform(-0.16, 0.16, len(pts))

    # 비유의(비움) 먼저 그려 채움 점이 위로 오게 한다
    for mask, face in ((~sig, "none"), (sig, color)):
        if mask.any():
            ax.scatter(x0 + jit[mask], pts[mask], s=52,
                       facecolors=face, edgecolors=color,
                       linewidths=1.6, zorder=3, clip_on=False)
            # 겹침 구간 가독성을 위한 흰 테두리(채움 점에만 의미가 있다)
            if face != "none":
                ax.scatter(x0 + jit[mask], pts[mask], s=52,
                           facecolors="none", edgecolors=SURFACE,
                           linewidths=0.6, zorder=4, clip_on=False)
    med = float(np.median(pts))
    ax.plot([x0 - 0.26, x0 + 0.26], [med, med], color=color, lw=2.2,
            solid_capstyle="butt", zorder=5)
    return med


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", default="figure_risk")
    ap.add_argument("--label", default=None, help="제목에 덧붙일 모델 이름")
    ap.add_argument("--null-name", default="귀무", help="CSV의 귀무 짝 표기")
    ap.add_argument("--cross-name", default="교차", help="CSV의 교차 짝 표기")
    ap.add_argument("--seed", type=int, default=0, help="흔들림 고정 시드")
    a = ap.parse_args()

    ko = pick_font()
    if ko:
        T = dict(yl="효과 크기 (%p)",
                 xnull="귀무 짝", xcross="교차 짝",
                 subnull="참 효과 = 0", subcross="ko vs en_instruct",
                 excl="0 배제", mx="최대",
                 sig="0을 배제(“유의”)", nsig="0을 포함",
                 ttl="단일 캘리브레이션 실행으로 얻은 효과 크기: 귀무 짝과 교차 짝",
                 fpr="위양성률", maxabs="최대 |값|",
                 note=("귀무 짝은 같은 언어로 캘리브레이션한 시드끼리의 비교이므로 참 효과가 0이다. "
                       "그럼에도 상당수가 0을 배제하며, 그 크기가 교차 짝과 겹친다."))
    else:
        T = dict(yl="Effect size (pp)",
                 xnull="Null pairs", xcross="Cross pairs",
                 subnull="true effect = 0", subcross="ko vs en_instruct",
                 excl="excl. 0", mx="max",
                 sig="Excludes 0 (“significant”)", nsig="Includes 0",
                 ttl="Effect sizes from a single calibration run: null vs cross pairs",
                 fpr="False-positive rate", maxabs="max |value|",
                 note=("Null pairs compare seeds calibrated in the same language, so the true "
                       "effect is 0. Many still exclude 0, at magnitudes overlapping the cross pairs."))

    rows, ns = load(a.csv)
    if not rows:
        print(f"CSV에서 읽을 행이 없다 → {a.csv}")
        return 1
    have = [q for q in QUANTS if (q, a.cross_name) in rows or (q, a.null_name) in rows]
    if not have:
        keys = sorted({k[0] for k in rows}), sorted({k[1] for k in rows})
        print(f"quantity/pair_type 값이 예상과 다르다. CSV에 있는 값: {keys}")
        print(f"--null-name / --cross-name 으로 맞춰 줄 것.")
        return 1

    rng = np.random.default_rng(a.seed)
    fig, axes = plt.subplots(1, len(have), figsize=(3.7 * len(have), 4.4), sharey=True)
    if len(have) == 1:
        axes = [axes]

    summary = []
    for ax, q in zip(axes, have):
        nullv = rows.get((q, a.null_name), [])
        crossv = rows.get((q, a.cross_name), [])
        ax.axhline(0, color="#a0aec0", lw=1, zorder=1)
        strip(ax, 0, nullv, C_NULL, rng)
        strip(ax, 1, crossv, C_CROSS, rng)

        fpr = 100 * sum(s for _, s in nullv) / len(nullv) if nullv else float("nan")
        cpr = 100 * sum(s for _, s in crossv) / len(crossv) if crossv else float("nan")
        mx_n = max((abs(v) for v, _ in nullv), default=float("nan"))
        mx_c = max((abs(v) for v, _ in crossv), default=float("nan"))
        summary.append((q, len(nullv), fpr, mx_n, len(crossv), cpr, mx_c))

        ax.set_title(q.replace("_", "$_{") + "}$" if q != "X" else "X",
                     fontsize=12, color=INK, pad=10)
        ax.set_xticks([0, 1])
        ax.set_xticklabels([
            f"{T['xnull']} · n={len(nullv)}\n{T['subnull']}\n"
            f"{T['excl']} {fpr:.1f}% · {T['mx']} {mx_n:.2f}",
            f"{T['xcross']} · n={len(crossv)}\n{T['subcross']}\n"
            f"{T['excl']} {cpr:.1f}% · {T['mx']} {mx_c:.2f}",
        ], fontsize=8.5, color=INK_MUTED, linespacing=1.5)
        ax.set_xlim(-0.55, 1.55)
        ax.grid(axis="y", alpha=.22, lw=.8)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color("#cbd5e0")
        ax.tick_params(colors=INK_MUTED, length=3)


    axes[0].set_ylabel(T["yl"], fontsize=10, color=INK)

    handles = [Line2D([], [], marker="o", ls="none", ms=8, mfc=INK_MUTED,
                      mec=INK_MUTED, label=T["sig"]),
               Line2D([], [], marker="o", ls="none", ms=8, mfc="none",
                      mec=INK_MUTED, mew=1.6, label=T["nsig"])]
    fig.legend(handles=handles, loc="upper right", frameon=False, fontsize=9,
               bbox_to_anchor=(0.995, 0.965), labelcolor=INK_MUTED)

    ttl = T["ttl"] + (f"  [{a.label}]" if a.label else "")
    fig.suptitle(ttl, fontsize=12, color=INK, y=0.99, x=0.02, ha="left")
    fig.text(0.02, 0.02, T["note"], fontsize=8.5, color=INK_MUTED, ha="left")
    fig.tight_layout(rect=(0, 0.07, 1, 0.92))

    for ext in ("png", "pdf"):
        fig.savefig(f"{a.out}.{ext}", dpi=200, bbox_inches="tight",
                    facecolor=SURFACE)

    print(f"저장: {a.out}.png / .pdf  (한글 폰트 {'사용' if ko else '없음 → 영문 라벨'})")
    print(f"  n 값: {sorted(ns)}")
    print(f"  {'지표':6s} {'귀무n':>5s} {'위양성률':>8s} {'귀무최대':>9s} "
          f"{'교차n':>5s} {'유의율':>7s} {'교차최대':>9s}")
    for q, n0, f0, m0, n1, f1, m1 in summary:
        print(f"  {q:6s} {n0:5d} {f0:7.1f}% {m0:9.2f} {n1:5d} {f1:6.1f}% {m1:9.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
