# ⚽ EPL Value Betting — End-to-End Prediction & Backtest Pipeline

Production-пайплайн: данные football-data.co.uk → leak-free фичи → XGBoost/LogReg →
value betting бэктест по Pinnacle closing odds → Walk-Forward → **честный вердикт**.

## Запуск

```bash
pip install -r requirements.txt
python3 run_pipeline.py          # данные → фичи → модели → бэктест → графики → отчёт
python3 dashboard/serve.py 8000  # web-дашборд: http://localhost:8000
```

## Структура

| Путь | Что это |
|---|---|
| `src/data.py` | Скачивание/очистка E0 за 5 сезонов (comment='#') |
| `src/features.py` | Elo (K=20, home adv=100), форма N=3/5, сезонные метрики, H2H |
| `src/models.py` | XGB (`multi:softprob`) + LogReg, GridSearchCV + **TimeSeriesSplit**, isotonic-калибровка (кросс-валидированная, без утечки) |
| `src/backtest.py` | Value betting: `model_prob > (1/odds)*(1+threshold)`, фикс-ставка 1% банка, банк 10 000 |
| `src/charts.py` | Reliability diagram, кривая банка, walk-forward бары |
| `dashboard/` | Web-дашборд (zero-dependency, stdlib http.server) |
| `outputs/` | `results.json`, `test_bets.csv`, `features_full.csv`, PNG |

## Анти-утечная дисциплина

- Признаки — **только информация до матча**: Elo до игры, форма по прошлым матчам, H2H до игры.
- Запрещённые колонки (`FTHG`, `FTAG`, `FTR`, `HS`, `AS`, `HST`, `AST`,…) в признаки не входят.
- Коэффициенты — только для бэктеста, не для обучения.
- Split строго хронологический: train 20/21–22/23 · val 23/24 · test 24/25.
- Выбор модели/порога — **только на validation**. Test трогается один раз.
- Калибровка — isotonic, кросс-валидированная внутри train (TimeSeriesSplit), чтобы validation оставалась чистой.

## Результаты (финальный Test 2024/25)

```
=== ВЕРДИКТ ===
Модель: XGBoost (multi:softprob, isotonic-calibrated)
Test season: 2024/25
ROI: -36.09%          (bank 10 000 → 6 391)
Yield: -13.80%
Количество ставок: 341
Max Drawdown: 41.99%
Baseline ROI (фаворит): -15.66%
Test accuracy: 52.1%   (Validation accuracy: 55.5%)
```

### Test по порогам edge

| Стратегия | Ставок | ROI | Yield | Win rate | Max DD |
|---|---|---|---|---|---|
| edge > 5% | 341 | -36.09% | -13.80% | 24.9% | 41.99% |
| edge > 10% | 298 | -32.02% | -13.47% | 23.8% | ~40% |

### Стабильность (Walk-Forward: окно 2 сезона → следующий)

| Train | Test | ROI | Yield | Ставок |
|---|---|---|---|---|
| 20/21+21/22 | 22/23 | +1.66% | +0.50% | 363 |
| 21/22+22/23 | 23/24 | -24.71% | -8.37% | 332 |
| 22/23+23/24 | 24/25 | -53.28% | -23.59% | 350 |

### Red Flags
- Модель **не бьёт** baseline «всегда на фаворит рынка» (-36.1% против -15.7%).
- ROI волатилен между WF-окнами (от +1.7% до -53.3%) — модель нестабильна.
- Ставится ~90% матчей по edge>5% — фильтр почти не режет (probabilities не отделены от маржи).

### Заключение
> ❌ Модель не демонстрирует устойчивого преимущества.
> **Не рекомендуется для реальных ставок.**

Closing odds Pinnacle — эффективный рынок: «edge» модели против market prob исчезает
после маржи. Никакими порогами результат не «улучшался» (запрещено промптом).

## Замечания по реальному применению
- Бэктест по closing odds завышает достижимый результат — в реале коэффициент до закрытия хуже.
- Для следующей итерации: сравнение с opening odds и/или Kelly вместо flat.
