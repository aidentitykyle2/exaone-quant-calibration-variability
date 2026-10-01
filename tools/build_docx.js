// paper_short.md → paper_short.docx (A4 2장 학술 논문 형식)
const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType,
  Table, TableRow, TableCell, WidthType, ShadingType, BorderStyle,
} = require("docx");

const FONT = "Noto Sans CJK KR";
const BODY = 20;      // half-points → 10pt
const SMALL = 18;     // 9pt

const P = (text, o = {}) => new Paragraph({
  alignment: o.align || AlignmentType.JUSTIFIED,
  spacing: { line: 272, before: o.before ?? 0, after: o.after ?? 66 },
  indent: o.indent,
  children: runs(text, o),
});

// **굵게** 와 *기울임* 을 런으로 분해한다
function runs(text, o = {}) {
  const out = [];
  const re = /(\*\*[^*]+\*\*|\*[^*]+\*)/g;
  let last = 0, m;
  const push = (t, bold, italics) => {
    if (!t) return;
    out.push(new TextRun({ text: t, bold, italics, font: FONT, size: o.size || BODY }));
  };
  while ((m = re.exec(text)) !== null) {
    push(text.slice(last, m.index));
    const tok = m[0];
    if (tok.startsWith("**")) push(tok.slice(2, -2), true);
    else push(tok.slice(1, -1), false, true);
    last = m.index + tok.length;
  }
  push(text.slice(last));
  return out.length ? out : [new TextRun({ text: "", font: FONT, size: o.size || BODY })];
}

const H = (text, level, o = {}) => new Paragraph({
  heading: level,
  spacing: { before: o.before ?? 165, after: o.after ?? 66 },
  children: [new TextRun({ text, bold: true, font: FONT, size: o.size || 22 })],
});

// 수식: 가운데 정렬, 기울임
const EQ = (text) => new Paragraph({
  alignment: AlignmentType.CENTER,
  spacing: { before: 60, after: 74 },
  children: [new TextRun({ text, italics: true, font: FONT, size: BODY })],
});

const TOTAL = 9360;   // A4 본문 폭(DXA)

function table(header, rows, widths) {
  const cell = (t, opts = {}) => new TableCell({
    width: { size: opts.w, type: WidthType.DXA },
    shading: opts.head ? { type: ShadingType.CLEAR, fill: "F1F3F5" } : undefined,
    margins: { top: 40, bottom: 40, left: 90, right: 90 },
    children: [new Paragraph({
      alignment: opts.first ? AlignmentType.LEFT : AlignmentType.CENTER,
      spacing: { before: 0, after: 0, line: 240 },
      children: runs(t, { size: SMALL }).map(r => r),
    })],
  });
  return new Table({
    columnWidths: widths,
    width: { size: TOTAL, type: WidthType.DXA },
    rows: [
      new TableRow({
        tableHeader: true,
        children: header.map((t, i) => cell(t, { w: widths[i], head: true, first: i === 0 })),
      }),
      ...rows.map(r => new TableRow({
        children: r.map((t, i) => cell(t, { w: widths[i], first: i === 0 })),
      })),
    ],
  });
}

const RULE = new Paragraph({
  spacing: { before: 60, after: 140 },
  border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: "CCCCCC" } },
  children: [new TextRun({ text: "", font: FONT, size: 2 })],
});

const kids = [];

// ── 표제
kids.push(new Paragraph({
  alignment: AlignmentType.CENTER,
  spacing: { after: 70 },
  children: [new TextRun({
    text: "4비트 LLM 양자화에서 측정된 언어 비대칭 효과의 신뢰성:",
    bold: true, font: FONT, size: 26 })],
}));
kids.push(new Paragraph({
  alignment: AlignmentType.CENTER,
  spacing: { after: 90 },
  children: [new TextRun({
    text: "캘리브레이션 표본 변동에 대한 분석", bold: true, font: FONT, size: 26 })],
}));
kids.push(new Paragraph({
  alignment: AlignmentType.CENTER,
  spacing: { after: 60 },
  children: [new TextRun({
    text: "How Reliable Are Measured Language Asymmetries in 4-Bit LLM Quantization?",
    font: FONT, size: SMALL, italics: true })],
}));
kids.push(new Paragraph({
  alignment: AlignmentType.CENTER,
  spacing: { after: 200 },
  children: [new TextRun({
    text: "An Analysis of Calibration Sampling Variability",
    font: FONT, size: SMALL, italics: true })],
}));

// ── 초록
kids.push(H("초록", HeadingLevel.HEADING_2, { before: 0, size: 22 }));
[
  "대규모 언어모델을 4비트로 압축할 때는 수백 개의 예시 문장을 모델에 보여 주고 그 반응을 관찰해 압축 방식을 정한다. 이 예시 문장이 한국어인지 영어인지가 압축 후 한국어 성능을 좌우한다는 보고가 있었으나, 그런 결론은 대개 **한 번의 실험**에서 나온다.",
  "예시 문장만 바꾸어 같은 실험을 반복하자 한국어·영어 손상 격차가 −2.45에서 +5.74 퍼센트포인트 사이에서 변했고, 18회 중 5회는 방향이 뒤집혔다. 한 번만 실험해 얻은 값(+3.27)은 18회 평균(+1.20)의 약 2.7배였다.",
  "더 중요한 확인은 다음과 같다. **어느 예시 문장이 한국어용이고 어느 것이 영어용인지 무작위로 뒤바꿔도 같은 크기의 “언어 격차”가 흔히 나타났다.** 모델 두 규모와 압축 기법 두 종류, 세 설정 모두에서, 두 언어의 대비를 함께 취한 지표는 예시 문장의 언어로 설명되었으나(*p* = 0.016 / 0.024 / 0.043) “한국어가 더 손상된다”는 한쪽 언어만의 값은 어느 설정에서도 설명되지 않았다(*p* = 0.171 / 0.054 / 0.113). 기존 연구가 보고해 온 양이 바로 후자다. 한 번의 실험에서 얻은 언어별 손상 차이를 그대로 결론으로 삼아서는 안 되며, 대상 언어만 평가하는 설계로는 그 차이를 확인할 수 없다.",
].forEach(t => kids.push(P(t, { indent: { left: 340, right: 340 }, size: SMALL })));
kids.push(P("**주제어**: LLM 양자화, 캘리브레이션 데이터, 표본 변동, 언어 비대칭, 평가 방법론",
  { indent: { left: 340, right: 340 }, size: SMALL, after: 60 }));
kids.push(RULE);

// ── 1
kids.push(H("1. 서론", HeadingLevel.HEADING_2));
kids.push(P("사후 학습 양자화(GPTQ, AWQ 등)는 캘리브레이션 데이터에서 활성화 통계를 관측해 양자화 파라미터를 정한다. 다국어 모델에서 이 데이터의 언어가 언어별 성능에 영향을 준다는 보고가 이어져 왔으나 결론이 엇갈린다."));
kids.push(P("대부분의 평가는 고정된 캘리브레이션 세트로 모델을 만들고 고정된 벤치마크로 비교하며, 평가 문항의 불확실성만 재표집으로 추정한다. 그러나 캘리브레이션 데이터 역시 큰 분포에서 뽑은 하나의 표본이고, 서로 다른 표본은 서로 다른 양자화 모델을 만든다. 관심 대상이 1~4 퍼센트포인트 규모의 차이일 때 이 문제는 결정적이다. **본 연구는 캘리브레이션 표본 선택이 측정된 언어 비대칭에 얼마나 영향을 미치며, 관측된 차이를 캘리브레이션 언어에 귀속할 수 있는지를 묻는다.**"));

// ── 2
kids.push(H("2. 실험 설계", HeadingLevel.HEADING_2));
kids.push(P("EXAONE 3.5-7.8B와 32B(리비전 고정)를 GPTQ·W4A16·그룹 128로 양자화하고 7.8B에서는 AWQ를 병행했다. 캘리브레이션은 한국어 KoAlpaca와 영어 Alpaca에서 각 256개를 뽑아 도메인을 instruction으로 맞췄다. 추출 시드를 바꾸어 7.8B GPTQ 18회, 7.8B AWQ 6회, 32B GPTQ 6회 반복했고 조건별 표본 집합의 해시가 모두 상이함을 확인했다. 평가는 KMMLU(35,030문항)와 MMLU(14,042문항) 5-shot이며, 선택지 로그우도 동률이 다수 발생하므로 동률에 점수를 나눠 주는 채점을 주 기준으로 쓴다."));
kids.push(P("BF16 대비 저하율과 언어 비대칭을 다음과 같이 정의한다."));
kids.push(EQ("D(c, B) = [Acc_BF16(B) − Acc_c(B)] / Acc_BF16(B) × 100"));
kids.push(EQ("L_KO = D(en, B_KO) − D(ko, B_KO),   L_EN = D(en, B_EN) − D(ko, B_EN)"));
kids.push(EQ("X = L_KO − L_EN"));
kids.push(P("가설이 예측하는 방향은 L_KO > 0, L_EN < 0, X > 0이다. 평가 문항 변동은 문항 재표집으로, 캘리브레이션 표본 변동은 표본을 바꾼 양자화 반복으로 추정하고, 두 변동의 상대 크기는 **표본 간 표준편차 대 문항 표준오차의 비**로 판단한다."));
kids.push(P("귀속 가능성은 **순열 검정**으로 판정한다. 각 모델이 갖는 (한국어 저하율, 영어 저하율) 쌍을 유지한 채 캘리브레이션 언어 라벨만 재배정하고 같은 통계량을 계산한다. 귀무가설은 “라벨이 교환 가능하다”이며 이 가설 아래 재배정은 정확한 귀무분포를 만든다. 조건당 6표본이면 924가지를 전수 열거하므로 근사가 없다."));

// ── 3
kids.push(H("3. 결과", HeadingLevel.HEADING_2));
kids.push(H("3.1 표본에 따라 측정값이 크게 달라진다", HeadingLevel.HEADING_3, { size: 20, before: 140 }));
kids.push(P("7.8B GPTQ 18회에서 시드별 X는 −2.45에서 +5.74에 분포했고 **5회에서 방향이 뒤바뀌었다.**"));
kids.push(table(
  ["불확실성 계산 방식", "L_KO", "L_EN", "X"],
  [["표본 하나, 문항만 재표집", "+1.27 [+0.49, +2.04]", "−2.01 [−2.73, −1.30]", "**+3.27 [+2.20, +4.33]**"],
   ["18표본, 문항+표본 재표집", "+0.52 [−0.27, +1.26]", "−0.66 [−1.22, −0.10]", "**+1.20 [+0.12, +2.16]**"]],
  [2700, 2220, 2220, 2220]));
kids.push(P("두 값은 추정 대상이 다르므로 2.7배라는 비를 편향의 크기로 읽을 수는 없다.", { before: 110 }));

kids.push(H("3.2 무엇을 캘리브레이션 언어에 귀속할 수 있는가", HeadingLevel.HEADING_3, { size: 20 }));
kids.push(table(
  ["설정", "Δ_KO", "Δ_EN", "X"],
  [["7.8B GPTQ (18×18)", "+0.50, p=0.171", "+0.66, **p=0.009**", "+1.16, **p=0.016**"],
   ["7.8B AWQ (6×6, 전수)", "+1.04, p=0.054", "+0.62, p=0.227", "+1.65, **p=0.024**"],
   ["32B GPTQ (6×6, 전수)", "+1.03, p=0.113", "+0.21, p=0.199", "+1.23, **p=0.043**"]],
  [2700, 2220, 2220, 2220]));
kids.push(P("**두 규모와 두 기법에서 같은 구조가 나타난다.** 두 평가 언어의 대비를 함께 취한 X는 세 설정 모두에서 캘리브레이션 언어에 귀속되는 반면, 한국어 쪽 단순 효과 Δ_KO는 어느 설정에서도 귀속되지 않는다. 세 설정 모두 Δ_KO의 귀무분포 95분위가 관측값 이상이다. X만 검출되는 것은 두 성분이 같은 부호여서 신호는 더해지는 반면 귀무 변동은 선형으로 더해지지 않기 때문이다(32B에서 독립 가정 시 0.654, 실측 0.618).", { before: 110 }));

kids.push(H("3.3 변동의 구조와 측정 절차의 영향", HeadingLevel.HEADING_3, { size: 20 }));
kids.push(P("캘리브레이션 표본 간 표준편차 대 문항 표준오차의 비는 **한국어 평가에서 2.7~3.8, 영어 평가에서 0.8~1.0**이다(32B). 보존하려는 대상 언어일수록 단일 실행의 위험이 크다. 캘리브레이션 크기를 128에서 1,024개로 늘려도 표본 간 표준편차는 1.0 내외로 유지되어 1/√n 감소가 관찰되지 않았다(각 크기 5회)."));
kids.push(P("채점 규칙도 결과를 바꾼다. 동률을 첫 선택으로 처리하면 32B의 X가 +2.30, 분할 처리하면 +1.23으로 **1.87배** 차이가 난다. 동률률은 규모와 함께 증가하며(BF16 KMMLU 24.9% → 36.4%) 두 규모 모두 BF16이 양자화 모델보다 높아, 첫 선택 채점에서는 4비트가 BF16보다 정확해 보이는 음의 저하율이 나타난다. 지표별 필요 표본 수도 달라 32B 한국어 평가에서 정확도는 7표본, 나머지 세 지표는 각 2표본을 요구했다."));

// ── 4
kids.push(H("4. 논의와 실무 권고", HeadingLevel.HEADING_2));
kids.push(P("측정된 비대칭을 부풀리는 요인이 서로 독립적으로 둘 있다. 단일 캘리브레이션 실행(2.7배)과 첫 선택 동률 채점(1.87배)이며, 겹치면 실제 +1 퍼센트포인트 수준의 효과가 +3 이상으로 보고될 수 있다."));
kids.push(P("**측정 대상의 선택이 반복 횟수만큼 중요하다.** 선행 연구가 보고해 온 양은 대개 한쪽 언어의 단순 효과이며, 세 설정 모두에서 그 양은 캘리브레이션 언어에 귀속되지 않았다. 대상 언어만 평가하는 설계는 귀속 불가능한 양을 측정하게 되므로, 대조 언어 벤치마크를 함께 평가하는 것은 보완재가 아니라 필수 조건에 가깝다. 이는 선행 연구의 결론이 엇갈리는 이유에 대한 한 가지 설명이기도 하다."));
kids.push(P("실무적으로는 두 평가 언어의 대비를 함께 취한 지표를 보고하고, 반복이 한 번 더 가능하다면 같은 언어의 다른 표본으로 만든 쌍을 먼저 확인할 것을 권한다. 거기서 관심 효과와 비슷한 크기가 나오면 그 설계로는 측정이 불가능하다. 정확도만 보지 말고, 동률 처리 방식과 캘리브레이션 데이터·표본 수·추출 시드·모델 리비전을 함께 보고한다."));

// ── 5
kids.push(H("5. 결론", HeadingLevel.HEADING_2));
kids.push(P("캘리브레이션 표본을 달리한 반복에서 측정된 언어 비대칭은 크기와 방향이 모두 표본에 민감했다. 라벨 재배정 순열 검정을 두 규모와 두 기법에 적용한 결과, **두 평가 언어의 대비를 함께 취한 지표는 세 설정 모두에서 캘리브레이션 언어에 귀속되었으나 한쪽 언어의 단순 효과는 어느 설정에서도 귀속되지 않았다.**"));
kids.push(P("본 연구는 언어 비대칭이 존재하지 않는다고 주장하지 않는다. 구간이 0을 포함하거나 검정이 유의하지 않다는 것은 효과의 부재가 아니라 방향을 확정할 증거가 부족하다는 뜻이다. 확인되는 것은 **단일 실행에서 얻은 크기를 양자화 기법이나 모델의 안정적 성질로 해석해서는 안 된다**는 점이다."));
kids.push(P("**한계.** 모델 계열이 EXAONE 3.5 하나이므로 규모 확장은 계열 간 일반화가 아니다. “한국어”는 KMMLU, “영어”는 MMLU이므로 언어 효과와 벤치마크 효과가 분리되지 않았다. 32B는 6표본으로 정확도 기준 검출에 필요한 7표본에 못 미친다. 진단과 설계 권고를 제시하나 같은 예산에서 변동을 줄인 완화 실험은 포함하지 않았다.", { size: SMALL }));
kids.push(P("**재현 정보.** 모델 리비전, 라이브러리 버전, 조건별 캘리브레이션 표본 목록과 집합 해시, 산출물 검증 절차, 분석 코드를 공개한다.", { size: SMALL }));

const doc = new Document({
  styles: { default: { document: { run: { font: FONT, size: BODY } } } },
  sections: [{
    properties: { page: { margin: { top: 960, bottom: 900, left: 1000, right: 1000 } } },
    children: kids,
  }],
});

Packer.toBuffer(doc).then(b => {
  fs.writeFileSync("paper_short.docx", b);
  console.log("paper_short.docx 작성 완료");
});
