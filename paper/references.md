# 참고문헌

번호는 **본문 첫 인용 순서**다. `paper_short.md`(제출 요약본)와 `paper.md`(전문)가
같은 번호를 쓴다. 논문집 양식은 대괄호 뒤를 띄우지 않는다(`[1]저자`).

## 목록

```
[1]M. Williams and N. Aletras, "On the Impact of Calibration Data in Post-training
Quantization and Pruning," Proc. ACL, pp. 10100–10118, 2024.

[2]E. A. Chimoto, M. Elhoushi, and B. A. Bassett, "Calibrating Beyond English:
Language Diversity for Better Quantized Multilingual LLMs," Proc. EACL,
pp. 4822–4838, 2026.

[3]K. Marchisio, S. Dash, H. Chen, D. Aumiller, A. Üstün, S. Hooker, and S. Ruder,
"How Does Quantization Affect Multilingual LLMs?," Findings of EMNLP,
pp. 15928–15947, 2024.

[4]K. A. K. Borgersen and M. Goodwin, "English K_Quantization of LLMs Does Not
Disproportionately Diminish Multilingual Performance," in Artificial Intelligence
XLII (SGAI-AI 2025), Lecture Notes in Computer Science, vol. 16301,
pp. 106–115, 2025.

[5]E. Frantar, S. Ashkboos, T. Hoefler, and D. Alistarh, "GPTQ: Accurate
Post-Training Quantization for Generative Pre-trained Transformers,"
Proc. ICLR, 2023.

[6]J. Lin, J. Tang, H. Tang, S. Yang, W.-M. Chen, W.-C. Wang, G. Xiao, X. Dang,
C. Gan, and S. Han, "AWQ: Activation-aware Weight Quantization for LLM
Compression and Acceleration," Proc. MLSys, 2024.

[7]G. Son, H. Lee, S. Kim, S. Kim, N. Muennighoff, T. Choi, C. Park, K. M. Yoo,
and S. Biderman, "KMMLU: Measuring Massive Multitask Language Understanding
in Korean," Proc. NAACL-HLT, pp. 4076–4104, 2025.

[8]D. Hendrycks, C. Burns, S. Basart, A. Zou, M. Mazeika, D. Song, and
J. Steinhardt, "Measuring Massive Multitask Language Understanding,"
Proc. ICLR, 2021.

[9]B. Efron and R. J. Tibshirani, "An Introduction to the Bootstrap,"
Chapman & Hall, 1993.

[10]P. Good, "Permutation, Parametric, and Bootstrap Tests of Hypotheses,"
3rd ed., Springer, 2005.
```

## 본문 인용 위치

| 번호 | 첫 인용 위치 | 역할 |
|---|---|---|
| [1]–[4] | 서론, 선행 연구 서술 문장 | 캘리브레이션 데이터·언어를 조작한 선행 연구. [4]는 반대 방향 결과 |
| [5] | 2.1, `GPTQ`로 W4A16 양자화 | 양자화 기법 |
| [6] | 2.1, `AWQ`를 병행 | 양자화 기법 |
| [7] | 2.1, 평가 벤치마크 `KMMLU` | 45개 과제·35,030문항 (논문 보고값) |
| [8] | 2.1, 평가 벤치마크 `MMLU` | 57개 과제. 문항 수는 논문에 없으므로 본 실험 집계값으로 표기 |
| [9] | 2.3, 문항 재표집 | 부트스트랩 |
| [10] | 2.3, 순열 검정 | 라벨 재배정 검정, 전수 열거 |

[2]와 [4]는 논의 절에서 "선행 연구의 결론이 엇갈린다"의 근거로 재인용한다.

## 번호 유지 조건

서론에서 **GPTQ·AWQ와 부트스트랩에 인용을 붙이지 않는다.** 붙이면 [5][6][9]가
[1]보다 먼저 등장해 전체를 다시 매겨야 한다. 두 경우 모두 관행을 기술한
일반적 언급이고, 해당 문헌은 2장에서 곧 인용된다.

## 문항 수의 출처

- KMMLU 35,030 — [7]이 보고한 값이며 본 실험 집계와 일치한다.
- MMLU 14,042 — **본 실험 집계값이다.** [8]의 초록은 과제 수 57개만 밝히고
  문항 수를 주지 않으며, lm-eval 버전에 따라 달라진다. 본문에서 `[8](본 실험
  14,042문항)` 형태로 출처를 분리해 적는다.

## BibTeX

```bibtex
@inproceedings{williams2024calibration,
  title     = {On the Impact of Calibration Data in Post-training Quantization and Pruning},
  author    = {Williams, Miles and Aletras, Nikolaos},
  booktitle = {Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics},
  pages     = {10100--10118},
  year      = {2024}
}

@inproceedings{chimoto2026calibrating,
  title     = {Calibrating Beyond English: Language Diversity for Better Quantized Multilingual LLMs},
  author    = {Chimoto, Everlyn Asiko and Elhoushi, Mostafa and Bassett, Bruce A.},
  booktitle = {Proceedings of the Conference of the European Chapter of the Association for Computational Linguistics},
  pages     = {4822--4838},
  year      = {2026}
}

@inproceedings{marchisio2024quantization,
  title     = {How Does Quantization Affect Multilingual LLMs?},
  author    = {Marchisio, Kelly and Dash, Saurabh and Chen, Hongyu and Aumiller, Dennis and
               {\"U}st{\"u}n, Ahmet and Hooker, Sara and Ruder, Sebastian},
  booktitle = {Findings of the Association for Computational Linguistics: EMNLP},
  pages     = {15928--15947},
  year      = {2024}
}

@incollection{borgersen2025english,
  title     = {English K\_Quantization of LLMs Does Not Disproportionately Diminish Multilingual Performance},
  author    = {Borgersen, Karl Audun Kagnes and Goodwin, Morten},
  booktitle = {Artificial Intelligence XLII},
  series    = {Lecture Notes in Computer Science},
  volume    = {16301},
  pages     = {106--115},
  year      = {2025},
  publisher = {Springer}
}

@inproceedings{frantar2023gptq,
  title     = {GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers},
  author    = {Frantar, Elias and Ashkboos, Saleh and Hoefler, Torsten and Alistarh, Dan},
  booktitle = {International Conference on Learning Representations},
  year      = {2023}
}

@inproceedings{lin2024awq,
  title     = {AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration},
  author    = {Lin, Ji and Tang, Jiaming and Tang, Haotian and Yang, Shang and Chen, Wei-Ming and
               Wang, Wei-Chen and Xiao, Guangxuan and Dang, Xingyu and Gan, Chuang and Han, Song},
  booktitle = {Proceedings of Machine Learning and Systems},
  year      = {2024}
}

@inproceedings{son2025kmmlu,
  title     = {KMMLU: Measuring Massive Multitask Language Understanding in Korean},
  author    = {Son, Guijin and Lee, Hanwool and Kim, Sungdong and Kim, Seungone and
               Muennighoff, Niklas and Choi, Taekyoon and Park, Cheonbok and
               Yoo, Kang Min and Biderman, Stella},
  booktitle = {Proceedings of the 2025 Conference of the Nations of the Americas Chapter of
               the Association for Computational Linguistics: Human Language Technologies},
  pages     = {4076--4104},
  year      = {2025}
}

@inproceedings{hendrycks2021measuring,
  title     = {Measuring Massive Multitask Language Understanding},
  author    = {Hendrycks, Dan and Burns, Collin and Basart, Steven and Zou, Andy and
               Mazeika, Mantas and Song, Dawn and Steinhardt, Jacob},
  booktitle = {International Conference on Learning Representations},
  year      = {2021}
}

@book{efron1993bootstrap,
  title     = {An Introduction to the Bootstrap},
  author    = {Efron, Bradley and Tibshirani, Robert J.},
  publisher = {Chapman \& Hall},
  year      = {1993}
}

@book{good2005permutation,
  title     = {Permutation, Parametric, and Bootstrap Tests of Hypotheses},
  author    = {Good, Phillip},
  edition   = {3rd},
  publisher = {Springer},
  year      = {2005}
}
```

## 제외한 문헌

초기 원고에는 QuaRot, SparseGPT, Wanda, SpQR 네 편이 있었다. 본문에서 인용되는
문장이 없어 제외했다. QuaRot의 R3·R4 미적용은 README의 "알려진 제약"에서 다루며
기법 자체를 논하지는 않는다.
