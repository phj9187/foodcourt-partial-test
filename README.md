# foodcourt-partial-test

[foodcourt-adaptive-sim](https://github.com/phj9187/foodcourt-adaptive-sim) 본 실험(`extended_main_20261108_001`)이 끝난 뒤, 일부 점수표만 따로 확인하거나 지정한 가중치 구간 안에서 새 점수표를 뽑아 학습·테스트하는 코드입니다.

- 본 실험과 같은 소스(`642cf41`), 같은 설정(`config/adaptive_extended.yaml`), 같은 `run_job`을 그대로 씁니다.
- 새 반복 번호는 새 시드를 뜻합니다(시드 = 설정 시드, 점수표 id, 반복 번호). 본 실험에 이미 있는 반복 번호는 거부합니다.
- 결과는 본 실험의 기존 반복과 합쳐 `partial_summary.csv` / `partial_summary.json`으로 정리합니다.

## Colab

[partial_test_colab.ipynb 열기](https://colab.research.google.com/github/phj9187/foodcourt-partial-test/blob/main/partial_test_colab.ipynb)

2번 셀에서 무엇을 돌릴지 정하고, 그 셀이 출력하는 가중치 목록을 확인한 뒤 3번 셀을 실행합니다.

## 명령줄

```bash
python partial_test.py --source <foodcourt-adaptive-sim 폴더> --output <새 폴더> \
  --reference <본 실험 폴더> \
  --objectives lhs_0022,lhs_0019 \
  --sample 8 --range "lambda=0.65..0.95,q=0.35..0.65,b=0..0.10,a_share=0..1" \
  --custom mine:0.2,0.05,0.75,0.8,0.5 \
  --reps 5-6 --workers $(nproc) [--dry-run | --aggregate-only]
```

| 기호 | 뜻 |
|---|---|
| a, b, c | 손님 점수 안의 맛, 가격 부담, 대기 가중치 (합 1) |
| λ (`lambda`) | 전체 점수 중 손님 점수의 몫 |
| q | 가게 점수 중 이익의 몫 (나머지는 폐기 줄이기) |
| `a_share` | 가격 부담 b를 뺀 나머지 중 맛의 비율, a/(a+c) |
