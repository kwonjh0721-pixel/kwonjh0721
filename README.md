# OIBC 2026: 규정 조건을 반영한 발전량 예측·집합자원 배분

친구가 제공한 Claude 모델의 **2024-06~2025-05 검증 CSV 전용 수정본**은
[claude_validation/README.md](claude_validation/README.md)를 참고하세요.
아래 설명은 기존 GPT 파이프라인에 관한 내용이며, 두 파이프라인의 검증 구간·채점 방식은 다릅니다.

제공된 `README.md`, `baseline.py`, 대회 페이지 캡처에 근거한 Python 파이프라인입니다.
전처리 → 시간순 검증 → 발전소 배분 최적화 → 최종 학습 → 제출 CSV 생성·검사까지 수행합니다.
기존 baseline의 태양 위치 계산을 참고했으며, 외부 데이터나 사전학습 모델은 사용하지 않습니다.

**현재 확인 범위:** 첨부된 실제 `sites.csv`와 `smp.csv`를 검사했습니다.
학습 발전량·기상예보 Parquet와 `train/smp.csv`는 첨부되지 않아 실제 대회 데이터 학습 및 점수 측정은 미실행입니다.
자동 테스트는 합성 데이터로 수행하며, 예측 정확도를 증명하는 자료가 아닙니다.
문제정의서 원문과 별도 공식 채점기는 제공되지 않았습니다. 제출 직전 원문의 최신 무효 조건·동점 규정을 확인하세요.

## 1. 파일 배치

원본 ZIP을 로컬에서 해제한 뒤 다음 구조로 배치합니다. `--data`로 원본 폴더를 직접 지정해도 됩니다.
대용량 파일을 이 채팅으로 다시 전송할 필요 없이 본인 PC/서버에서 실행할 수 있습니다.

```text
oibc2026.py
model_backends.py
requirements.txt
data/
  sites.csv                       # 첨부본 포함
  smp.csv                         # 첨부본 포함: 평가 시간축
  train/
    smp.csv                       # 평가용 smp.csv와 다른 파일!
    generation.parquet
    forecast_wa.parquet
    forecast_wb.parquet
  test/
    forecast_wa.parquet
    forecast_wb.parquet
```

`observed_wa.parquet`, `observed_wb.parquet`, `.DS_Store`는 필요하지 않습니다.
파일이 없으면 학습 전에 누락 목록을 출력하고 종료합니다. 가짜 데이터나 0 예측으로 제출물을 대체하지 않습니다.

## 2. 설치 및 실행

아래 명령은 저장소 루트(`oibc2026.py`가 있는 폴더)에서 실행합니다.
Python **3.11 이상**이 필요하며, 아래 설치 예시는 검증에 사용한 Python 3.12 기준입니다.
전체 데이터의 메모리 사용량·소요 시간은 아직 측정하지 않았습니다. 큰 결합 행렬이 생성되므로 RAM 32GB 환경을 권장합니다.
추론은 기본 168시간 단위로 나누어 메모리 사용을 줄입니다. 학습 단계는 기간 전체의 특징 행렬을 메모리에 올립니다.

macOS(Homebrew가 설치된 환경):

```bash
brew install python@3.12 libomp
python3.12 -m venv .venv
source .venv/bin/activate
```

`libomp`는 LightGBM의 OpenMP 실행 라이브러리입니다. Python 패키지만 설치하면
`Library not loaded: @rpath/libomp.dylib` 오류가 날 수 있습니다.
설치 근거: [Homebrew Python 3.12](https://formulae.brew.sh/formula/python@3.12),
[Homebrew libomp](https://formulae.brew.sh/formula/libomp),
[LightGBM 설치 안내](https://lightgbm.readthedocs.io/en/stable/Installation-Guide.html).

Linux(Python 3.12와 venv가 설치된 환경):

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

Windows PowerShell(Python 3.12가 설치된 환경):

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
```

가상환경 활성화 후 공통 설치·실행 순서:

```bash
python --version
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -c "import numpy, pandas, pyarrow, lightgbm, sklearn; print('Dependencies OK')"

# 1) 파일 배치와 메타데이터 검사: missing_files가 []인지 확인
python oibc2026.py audit --data data

# 2) 초기 학습 → 배분 → 홀드아웃 검증 → 최종 학습 → 제출 파일 생성
python oibc2026.py run --data data --output outputs --threads 4

# 3) 저장된 홀드아웃 예측의 성능 평가: 재학습하지 않음
python oibc2026.py evaluate --data data --output outputs

# 4) 제출 CSV 형식·용량·시간축 검사: 예측 정확도 평가는 아님
python oibc2026.py validate --data data --output outputs
```

각 명령이 성공한 뒤 다음 명령을 실행하세요. `audit`에서 누락 파일이 있으면 종료 코드는 2입니다.
`audit`은 파일 존재 여부와 `sites.csv`·평가용 `smp.csv`를 검사하며,
대용량 Parquet의 모든 내용과 학습 가능성까지 보장하지는 않습니다. 실제 스키마·중복·시간 조건은 `run`에서 추가 검사합니다.
원본 데이터를 다른 위치에 두었다면 **모든 명령의 `--data data`를 동일한 원본 폴더 경로로 변경**하세요.
예: `--data "/Users/journey/Downloads/OIBC2026_dataset"`.
루트의 `smp.csv`와 `train/smp.csv`는 기간과 역할이 다르므로 서로 복사해 대체하면 안 됩니다.

`outputs`는 없거나 비어 있어야 합니다. 재실험에는 `--output outputs_v2`처럼 새 경로를 사용하세요.
설치 시 패키지를 받는 것 이외에 학습·추론 코드는 네트워크/API를 호출하지 않습니다.

실행 중 터미널에는 `1/5`부터 `5/5`까지 단계가 표시됩니다. 실데이터 소요 시간은 아직 측정하지 않았습니다.
추론 메모리가 부족하면 `run`에 `--chunk-hours 24`를 추가할 수 있습니다.
이 옵션은 학습 행렬 크기를 줄이지 않으며, `--threads`도 학습 메모리의 상한을 보장하지 않습니다.

저장된 모델과 배정을 이용한 재추론:

```bash
python oibc2026.py predict --data data --model-dir outputs --output outputs_repredict
```

재추론은 발전량이나 학습 예보를 읽지 않습니다. 최초 학습 당시 메타데이터·평가 시간축·평가 예보의 SHA-256과 일치하는지 확인합니다.
새 데이터셋에 임의로 재사용하는 기능은 아닙니다.

## 3. 모델과 전처리

- 기본 모델은 **LightGBM**이며, `--model catboost`, `--model xgboost`, `--model histgb`로 바꿀 수 있습니다.
  모델 선택·설치는 [9절](#9-대안-모델-선택과-비교)을 참고하세요.
  모든 모델은 발전소별 설비이용률(`energy / capacity_kw`)을 예측한 뒤 설비용량을 곱해 kWh로 환산합니다.
- 전날 **10시·17시 두 라운드** 예보를 같은 행의 서로 다른 특징으로 넣습니다. 둘 다 사용 가능하다는 제공 자료에 근거합니다.
- A/B 예보 차이, 10→17시 예보 변화, 기온·운량·습도·풍속·강수·UV 등, 익명 발전소 ID와 제공 메타데이터를 사용합니다.
- B 풍속은 km/h에서 m/s로 환산합니다. 백분율 특징은 0~100으로 제한하고, 물리적으로 음수가 불가능한 선택 변수의 음수는 결측 처리합니다.
- 예보 누락은 NaN과 누락 표시 변수로 처리하며 LightGBM의 결측 처리를 사용합니다. 미래 시각의 값으로 보간하거나 역방향 채우기하지 않습니다.
- 제공된 익명 좌표와 시간으로 구간 중앙의 태양 위치·계절성을 계산합니다. 좌표 역추적, 외부 지도·기상 조회는 없습니다.
- 태양 고도가 -6도 이하이면 예측을 0으로 둡니다. 일출·일몰 주변을 남기기 위한 모델링 선택이며 대회 규정은 아닙니다.
- 발전량 결측 행은 **0으로 채우지 않습니다**. 유효한 라벨만 학습하고, 음수·비유한 라벨의 제거 건수를 기록합니다. 양수 이상치는 임의로 잘라내지 않습니다.
- 학습 손실은 SMP × k × 설비용량 가중 L1입니다. 이는 발전소별 오차의 대용 목적함수이며 집합 정산식과 완전히 같지는 않습니다. 배분 탐색과 검증에는 집합 정산식을 사용합니다.

## 4. 시간순 분리와 누수 방지

모든 시각은 KST, 우측닫힘 1시간 구간입니다. **6월 2일 00:00은 6월 1일의 마지막 시간**입니다.
대상일은 `(ts - 1시간).날짜`로 계산합니다. 학습 라벨은 첫 예측 대상일의 전날 17시 이하로 제한합니다.

기본 분리는 아래와 같습니다. 날짜 범위는 발전 대상일 기준입니다.

| 단계 | 사용하는 라벨/기간 | 용도 |
|---|---|---|
| 초기 학습 | 2023-06-01부터 2024-05-31 17:00까지 | 첫 모델 학습 |
| 튜닝·배분 | 2024-06-01부터 2025-02-28 17:00까지 | 조기종료 반복 수 선택, 집합 배분 탐색 |
| 홀드아웃용 재학습 | 2023-06-01부터 2025-02-28 17:00까지 | 반복 수·배정을 고정하고 모델 재학습 |
| 독립 홀드아웃 | 2025-03-01 01:00~2025-05-31 17:00 | 고정된 배정과 참조 배정의 성능 비교 |
| 최종 학습 | 2023-06-01부터 **2025-05-31 17:00**까지 | 첫 평가일 예측 시점에 존재하는 라벨만 이용 |
| 평가 예측 | 2025-06-01~2026-05-31 대상일 | 고정 모델·고정 배정으로 전체 평가 구간 예측 |

튜닝 기간 결과는 반복 수와 배정을 선택하는 데 사용되므로 독립 성능 추정치가 아닙니다.
홀드아웃 결과를 보고 자동으로 배정을 다시 선택하지 않습니다. 선택 배정이 홀드아웃에서 나빠질 수도 있습니다.
최종 모델은 홀드아웃 기간도 학습에 사용하므로, 저장된 홀드아웃 점수는 별도로 학습한 이전 모델의 점수입니다.
최종 모델 자체의 평가 구간 수익은 실제 발전량이 없어 계산할 수 없습니다.
제공된 학습 파일의 마지막 7시간(2025-05-31 18:00~2025-06-01 00:00)은 첫 평가 예측 시점보다 늦으므로
보수적으로 학습뿐 아니라 로컬 홀드아웃 채점에서도 제외합니다.

현재 구현은 하나의 시간순 홀드아웃입니다. 필요하면 다음 옵션으로 다른 시간 구간을 별도 실행할 수 있습니다.
홀드아웃을 반복 확인하며 튜닝하면 더 이상 독립 검증으로 볼 수 없다는 점에 유의하세요.

```bash
python oibc2026.py run --data data --output outputs_other_period \
  --train-from 2022-06-01 --optimize-from 2023-06-01 --holdout-from 2024-06-01
```

학습은 `train/`만 읽고, 평가 예보는 최종 추론에서만 읽습니다.
평가 SMP는 제출 시각을 정하는 데만 쓰며 모델 입력·튜닝 가중치로 사용하지 않습니다.
실측 기상과 발전량 lag는 학습·추론 특징에 포함하지 않아 평가 시 제공되지 않는 입력에 의존하지 않습니다.

## 5. 집합 배분과 수익 계산

253개 발전소를 정확히 한 집합에 배정하며 G1/G2 각각 20,000kW 이상을 유지합니다.
제공된 참조 배정은 A 20,465.655kW, B 22,189.715kW로 두 집합 모두 유효합니다.

튜닝 기간의 발전소별 예측 잔차를 이용하여, 발전소 단일 이동·두 발전소 교환 중 정산 오차를 줄이는 경우만 채택합니다.
매 변경에서 양쪽 용량을 검사합니다. 난수 시드는 고정이며 전역 최적해를 보장하는 방식은 아닙니다.
배정은 전체 평가 기간 동안 고정됩니다.

```text
총 매출 = Σ_t SMP(t) × [G1 실적(t) + G2 실적(t)]
패널티 = 0.10 × Σ_t SMP(t) × k(t)
                     × [|G1 예측(t) - G1 실적(t)| + |G2 예측(t) - G2 실적(t)|]
총 정산수익 = 총 매출 - 패널티
k(t) = ts의 시가 11~15이면 3, 그 외 1
```

모든 발전소를 배분하면 총 매출은 배정과 무관하므로 배분 탐색은 패널티를 최소화합니다.
발전량이 누락된 시각에 집합 실적을 계산하면 잘못된 점수가 나옵니다. 따라서 **253개 발전소 실적이 모두 있는 시각만** 탐색·점수 계산에 사용하고,
`complete_hours`, `total_hours`, `coverage`를 함께 기록합니다. 이 점수는 해당 시각 부분집합의 원화 정산액이며 연간 수익으로 환산하지 않습니다.
완전한 시각이 없으면 점수는 `null`입니다. 배분 탐색에 필요한 완전 시각이 기본 168개 미만이거나 가중치가 모두 0이면 참조 배정을 유지하고 사유를 기록합니다.

## 6. 산출물 및 제출 전 확인

| 파일 | 용도 |
|---|---|
| `composition.csv` | 제출용: 253행, `site_id,group` |
| `forecast.csv` | 제출용: 17,520행, `group,ts,forecast_kwh` |
| `model.txt` / `model.cbm` / `model.ubj` / `model.joblib` | 선택한 모델에 따라 하나 생성: LightGBM / CatBoost / XGBoost / HistGB |
| `site_predictions.parquet` | 평가 구간 발전소별 예측 |
| `holdout_site_predictions.parquet` | 독립 홀드아웃 발전소별 예측 |
| `validation_report.json` | 참조/선택 배정의 로컬 정산액, 실적 완전성, 제출 검사 결과 |
| `manifest.json` | 모델 종류·저장 형식, 파라미터, 실제 학습 시점, 입력·모델·홀드아웃 산출물 SHA-256, 패키지 버전, 특징 목록 |
| `feature_importance.csv` | LightGBM·CatBoost·XGBoost의 최종 모델 중요도; HistGB는 생성하지 않음 |
| `holdout_evaluation.json` | 선택 산출물: `evaluate --report`로 저장한 로컬 상세 평가 |

CSV 행 수는 헤더 제외입니다. 헤더를 포함하면 각각 254행, 17,521행입니다.
`validate`는 253개 발전소 누락·중복, 각 집합의 용량, G1/G2 명칭, 그룹별 8,760개 정확한 시각,
타임존, 키 중복, 음수·NaN·무한대 예측을 검사합니다.
자동 제출은 하지 않습니다. 화면 안내의 일 3회·전체 20회 제출 제한을 직접 관리하세요.
상위 7팀은 코드와 발표자료 제출 대상이며 안내된 마감은 2026-11-23입니다.
데이터는 대회 참가 목적으로만 사용하세요.

## 7. 성능 평가와 결과 해석

`run`이 완료되면 다음 명령으로 저장된 홀드아웃 예측을 다시 채점합니다.
모델을 재학습하지 않으며, `test/` 예보나 `model.txt`를 읽지 않습니다.

```bash
python oibc2026.py evaluate --data data --output outputs

# 결과를 JSON으로도 보관하려면 사용합니다. 기존 파일은 덮어쓰지 않습니다.
python oibc2026.py evaluate --data data --output outputs --report outputs/holdout_evaluation.json
```

필요한 파일은 원본의 `sites.csv`, `train/smp.csv`, `train/generation.parquet`와
실험 폴더의 `manifest.json`, `composition.csv`, `holdout_site_predictions.parquet`입니다.
`validation_report.json`이 있으면 배분 탐색 생략 사유도 표시합니다.
원본 데이터와 저장된 예측·배정의 SHA-256을 확인해 서로 다른 실험이나 변경된 파일이 섞이는 것을 탐지합니다.
이전 버전으로 만든 실험에는 산출물 해시가 없을 수 있으며, 이 경우 출처 검증 불가 안내를 표시하고 채점합니다.
임의로 교체한 예측 파일을 독립 홀드아웃 결과로 해석해서는 안 됩니다.

기본 평가 구간은 **2025-03-01 01시~2025-05-31 17시 KST**입니다.
기간을 변경해 학습했다면 해당 실험의 `manifest.json`에서 평가 구간을 읽습니다.
`Reference`는 제공된 A/B 배정, `Selected`는 탐색으로 선택한 G1/G2 배정입니다.
**동일한 홀드아웃 모델의 발전소별 예측을 두 배정으로 합산해 비교**하므로,
이 비교는 모델 교체 효과가 아닌 집합 배분 효과를 측정합니다.

| 표시 항목 | 의미 | 해석 |
|---|---|---|
| `S: settlement (KRW)` / `score_won` | 매전수익 − 오차 페널티 | 클수록 좋음 |
| `W: penalty (KRW)` / `penalty_won` | SMP·시간대 가중 집합 오차의 10% | 작을수록 좋음 |
| `Group MAE (kWh)` | 두 집합×채점 시간의 절대오차 평균 | 작을수록 좋음; 가중치 없는 보조 지표 |
| `Group RMSE (kWh)` | 두 집합×채점 시간의 제곱오차 평균의 제곱근 | 큰 오차에 더 민감한 보조 지표 |
| `Weighted WAPE` / `wwape` | Σ SMP×k×집합 절대오차 ÷ Σ SMP×k×집합 실적 | 작을수록 좋음; 화면은 %, JSON은 비율 |
| `Penalty reduction` | 참조 배정 W − 선택 배정 W | 양수면 배분 개선, 음수면 악화; 같은 시각의 S 증가액과 같음 |
| `Complete actual hours` / `coverage` | 모든 발전소의 유효 실적이 있는 시간 수/전체 시간 수 | 낮으면 평가 표본이 제한됨 |

모든 지표는 **같은 완전 실적 시각만** 사용합니다. 완전 실적 시각이 0개면 `N/A`/`null`을 출력하며,
좋은 점수인 0으로 대신하지 않습니다. WAPE의 실적 분모나 개선율의 참조 페널티가 0인 경우에도 해당 비율은 `null`입니다.
낮은 `coverage`에서는 일부 시각의 수익을 연간 수익으로 해석하거나, 다른 시간 범위의 실험과 금액만 비교하면 안 됩니다.
`Allocation search retained reference groups` 안내가 있으면 튜닝 기간의 완전 실적 시각 부족 등의 이유로 배분 탐색이 생략된 것입니다.
선택 배정이 홀드아웃에서 악화되었다고 해서 평가 명령이 자동으로 배정이나 모델을 바꾸지는 않습니다.

**`validate`와 `evaluate`의 역할은 다릅니다.** `validate` 통과는 제출 형식이 정상이라는 뜻이며,
`evaluate`는 이전 홀드아웃 모델의 로컬 성능을 측정합니다. 최종 모델은 이후 홀드아웃까지 포함해 재학습되므로
최종 저장 모델로 홀드아웃을 다시 예측하면 독립 검증이 아닙니다. 저장된 `holdout_site_predictions.parquet`를 사용하세요.

**공식 평가 구간(2025-06~2026-05)의 실제 발전량은 제공되지 않아 로컬에서 공식 점수를 계산할 수 없습니다.**
`validate` 통과 후 `composition.csv`와 `forecast.csv`를 대회 페이지에 함께 제출해 리더보드의 정산금 S를 확인합니다.
모델·Parquet·로컬 평가 JSON은 예측 CSV 제출물이 아닙니다. 제출 횟수는 일 3회, 전체 20회 제한을 따릅니다.
현재 로컬 정산식은 README의 식을 구현한 것이며, 별도 공식 채점기와의 동등성 검증은 아직 수행하지 못했습니다.

## 8. 코드 검증

```bash
python -m pip install pytest==8.3.5
python -m pytest -q tests claude_validation/tests
```

테스트에는 예측 시점 차단, 자정의 대상일 처리, 미래 예보 불변성, 누락 예보·발전량 처리,
정산식, 배분 용량 제약, 잘못된 제출물 거부, 합성 데이터 학습→저장→재추론 재현성이 포함됩니다.
저장된 홀드아웃 재채점, 누락 예측 거부, 데이터·산출물 변경 탐지, 채점 불가 상태 처리도 검사합니다.
테스트용 소규모 조건은 테스트 내부에서만 적용되며 실제 CLI의 253개/8,760시각 검사는 유지됩니다.

대안 모델까지 모두 검사하려면 아래 선택 의존성을 설치한 뒤 위 테스트를 실행합니다.
CatBoost·XGBoost가 설치되지 않았으면 해당 모델 테스트만 건너뛰며, LightGBM·HistGB 테스트는 실행합니다.

```bash
python -m pip install -r requirements-catboost.txt -r requirements-xgboost.txt
python -m pytest -q tests claude_validation/tests
```

## 9. 대안 모델 선택과 비교

추가 비교 우선순위는 **CatBoost → XGBoost → HistGradientBoosting**입니다.
이는 예보·발전소 메타데이터로 구성된 현재 표 형식 입력과 설치 비용을 고려한 실험 순서입니다.
실데이터 점수에 근거한 순위가 아니며, 어느 모델이 LightGBM보다 좋을지는 아직 확인하지 못했습니다.

| `--model` | 사용하는 모델과 목적함수 | 비교 이유 | 설치 |
|---|---|---|---|
| `lightgbm` (기본) | LGBMRegressor, L1 | 기존 기준 모델 | `requirements.txt` |
| `catboost` | CatBoostRegressor, MAE | 발전소 ID의 범주형 처리와 수치 결측 처리 지원; 다른 오차 패턴을 얻을 후보 | `requirements-catboost.txt` |
| `xgboost` | native XGBoost Booster, `reg:absoluteerror`, CPU histogram | L1·가중치·범주형 입력을 지원하는 별도 트리 부스팅 구현 | `requirements-xgboost.txt` |
| `histgb` | sklearn HistGradientBoostingRegressor, `absolute_error` | 기존 scikit-learn만으로 비교 가능; NaN·범주형 입력 지원 | 추가 설치 없음 |

기능 근거: [CatBoost 범주형 학습 입력](https://catboost.ai/docs/en/concepts/python-reference_catboostregressor_fit),
[CatBoost 결측 처리](https://catboost.ai/docs/en/concepts/algorithm-missing-values-processing),
[XGBoost 목적함수](https://xgboost.readthedocs.io/en/release_3.0.0/parameter.html),
[XGBoost 범주형 처리](https://xgboost.readthedocs.io/en/release_3.0.0/tutorials/categorical.html),
[sklearn 1.8 HistGradientBoostingRegressor](https://scikit-learn.org/1.8/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html).

LSTM·Transformer처럼 시퀀스 입력 구성이 필요한 모델은 후속 후보입니다. 현재는 시퀀스 구성·튜닝 비용 없이
같은 특징과 같은 시간 분리로 비교 가능한 위 모델부터 실험합니다. 외부 데이터로 학습된 사전학습 모델은 사용하지 않습니다.

### 실행

기존 `requirements.txt`와 `--model`을 생략한 명령은 계속 LightGBM을 사용합니다.
선택 라이브러리는 해당 모델을 선택할 때만 불러옵니다. CatBoost·XGBoost 설치 파일은 기존 요구사항도 포함합니다.
한 모델이 설치되지 않았을 때 다른 모델로 조용히 바꾸지 않고 설치 명령과 함께 오류를 출력합니다.

```bash
# 기존 기본 실행
python oibc2026.py run --data data --output outputs_lgbm --threads 4

# CatBoost
python -m pip install -r requirements-catboost.txt
python oibc2026.py run --data data --output outputs_catboost --model catboost --threads 4

# XGBoost
python -m pip install -r requirements-xgboost.txt
python oibc2026.py run --data data --output outputs_xgboost --model xgboost --threads 4

# scikit-learn HistGradientBoosting: 기본 의존성만 필요
python oibc2026.py run --data data --output outputs_histgb --model histgb --threads 4
```

처음에는 한 번에 한 모델씩 실행하세요. 각 실행은 자체적인 초기 학습·트리 수 선택·배분 최적화·홀드아웃 검증·최종 학습을
수행하며 별도 출력 폴더가 필요합니다. 다른 모델의 트리 수나 배정을 자동으로 재사용하지 않습니다.

```bash
# 모델 종류와 관계없이 같은 평가·제출 검사 명령 사용
python oibc2026.py evaluate --data data --output outputs_catboost
python oibc2026.py validate --data data --output outputs_catboost

# manifest에서 모델 종류와 특징/범주 정보를 읽어 자동으로 복원
python oibc2026.py predict --data data --model-dir outputs_catboost --output outputs_catboost_repredict
```

`predict`에는 `--model`을 다시 지정하지 않습니다. 학습 때 기록한 종류로 불러오며, 모델 파일 해시를 확인합니다.
기존 버전에서 만든 LightGBM `model.txt`와 `manifest.json`도 재추론할 수 있습니다.
HistGB의 `model.joblib`는 본인이 생성한 신뢰할 수 있는 파일만 사용하고, 학습 때와 같은 scikit-learn 버전이 필요합니다.

### 공통 조건과 모델별 차이

- 특징 생성, 설비이용률 목표, `SMP × k × 설비용량` 학습 가중치, 예측 후 비음수·야간 처리, 배분 탐색·채점식은 공통입니다.
- 초기 모델의 반복 수는 기존 튜닝 기간만으로 선택합니다. 모든 모델에 해당 기간의 검증 가중치를 전달합니다.
  홀드아웃·최종 재학습에서는 선택한 반복 수를 고정하고 조기종료를 끕니다.
- HistGB는 무작위 검증 분할 대신 명시적인 시간순 `X_val/y_val/sample_weight_val`을 받습니다.
  조기종료 후 최선의 반복 수가 마지막 반복 수와 다르면 초기 학습 구간만으로 한 번 더 학습합니다.
- CatBoost에는 발전소 ID를 문자열 범주로 전달합니다. MAE의 리프 추정은 `Gradient`로 명시합니다.
  고정한 1.2.8 버전의 작은 가중치 검증 사례에서 기본 `Exact`가 기대 방향으로 반응하지 않아 이 설정을 선택했습니다.
  이는 합성 사례에 대한 조치이며 실제 대회 성능 개선을 입증한 것은 아닙니다.
- XGBoost는 학습 구간에서 관측한 발전소 ID의 범주 목록을 저장합니다. 추론 묶음마다 ID를 다시 인코딩하지 않으며,
  학습에 없던 발전소 ID는 미지 범주(결측)로 처리합니다. 행을 버리지 않고 다른 메타데이터·예보 특징으로 예측합니다.
  scikit-learn 래퍼의 저장 호환성에 의존하지 않도록 native `train/Booster` API와 UBJSON 저장 형식을 사용합니다.
- HistGB는 `site_code`를 범주형으로 처리하고 `max_bins=255`를 사용해 253개 발전소를 지원합니다.
  스레드 수는 학습·추론의 `threadpoolctl` 범위 안에서 제한합니다.
- 중요도는 각 라이브러리의 자체 지표입니다. LightGBM은 기존 `feature,gain` CSV 형식을 유지하고,
  CatBoost·XGBoost는 `feature,importance`를 저장합니다. 종류는 manifest의 `importance_type`에 기록합니다.
  모델 간 중요도 숫자를 직접 비교하지 마세요. HistGB에는 같은 방식의 기본 중요도가 없어 파일을 만들지 않습니다.

| 설정 | 적용 모델 |
|---|---|
| `--trees`, `--early-stopping`, `--learning-rate`, `--threads`, `--seed` | 모든 모델 |
| `--num-leaves`, `--min-child-samples` | LightGBM, HistGB |
| `--max-depth` | CatBoost, XGBoost |
| 날짜 분리·`--chunk-hours`·배분 탐색 옵션 | 모든 모델의 공통 파이프라인 |

기본 학습률은 0.04, CatBoost·XGBoost의 기본 깊이는 6입니다.
모델별 옵션의 의미가 달라 `min_child_samples`를 XGBoost의 `min_child_weight` 등으로 억지로 대응시키지 않습니다.

### 비교와 확장

모델 자체를 비교하려면 같은 데이터·같은 날짜 분리에서 각 `evaluate` 결과의 **Reference W**를 비교하세요.
고정된 참조 배정에서 예측 모델만 달라집니다. 모델과 배분을 함께 비교하려면 **Selected W/S**를 비교합니다.
이때 평가 시간 범위와 `coverage`가 같은지 확인해야 합니다. W는 작을수록, S는 클수록 좋습니다.
여러 모델의 홀드아웃 점수를 보고 최종 모델까지 선택했다면 그 홀드아웃은 모델 선택에 사용된 구간이므로,
독립 성능 보고에는 별도의 미사용 기간이나 추가 시간순 검증이 필요합니다. 모델 선택과 앙상블은 자동으로 수행하지 않습니다.

추가 모델은 `model_backends.py`의 `RegressionModel` 계약을 구현하고 `BACKENDS`에 등록합니다.
`fit`은 학습·검증 가중치와 시간 분리를 지켜야 하며, `best_iteration_`은 인덱스가 아닌 실제 반복 수입니다.
`predict`, `save/load`, 특징 순서·범주 복원, 선택적 `importance`를 구현하면 기존 전처리·배분·제출 코드를 재사용할 수 있습니다.
원시 모델은 어댑터의 `.estimator`에 있으며, 실행에는 `oibc2026.py`와 `model_backends.py` 두 파일이 모두 필요합니다.
