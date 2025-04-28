# AI_stock_market_prediction

## Opis
Porównanie skuteczności różnych metod sztucznej inteligencji w prognozowaniu wahań rynku walutowego. Celem projektu jest analiza i ocena jakości predykcji modeli takich jak LSTM, XGBoost oraz ARIMA na bazie danych historycznych (2015–2020) i prognoz na 2021 rok.

## Wymagania
- Python 3.12.3
- biblioteki: pandas, numpy, scikit-learn, xgboost, tensorflow, keras, statsmodels, matplotlib, seaborn


## Instalacja

1. Sklonuj repozytorium:
   ```bash
   https://github.com/n4ce1/AI_stock_market_prediction.git
   AI_stock_market_prediction

## 🚀 Roadmap

### 🛠️ Przygotowanie środowiska lokalnego
- Zainstalowanie Python 3.12.3
- Zainstalowanie zależności
- Konfiguracja środowiska wirtualnego (np. venv)
- Utworzenie pliku `requirements.txt`

### 📊 Zebranie i obróbka danych
- Pozyskanie danych historycznych rynku walutowego (np. EUR/USD) za lata 2015–2020
- Wykonanie eksploracyjnej analizy danych (EDA)
- Obróbka danych (przekształcenie na szereg czasowy, normalizacja)
- Podział danych na zbiór treningowy i testowy

### 🤖 Trenowanie modeli
- Model ARIMA: Dobór parametrów, wytrenowanie modelu na danych 2015–2020, prognoza na 2021
- Model XGBoost: Przygotowanie cech wejściowych, wytrenowanie modelu, prognoza na 2021
- Model LSTM: Przygotowanie sekwencji wejściowych, budowa sieci neuronowej, wytrenowanie modelu, prognoza na 2021

### 📈 Ocena i porównanie modeli
- Wygenerowanie prognoz z każdego modelu na danych testowych
- Obliczenie metryk błędu (RMSE, MAPE)
- Pomiar czasu działania
- Ocena interpretowalności modeli

### 🔍 Wnioski i rekomendacje
- Analiza wyników i ocena najlepszych modeli
- Zidentyfikowanie potencjalnych słabości
- Sformułowanie rekomendacji do przyszłych prac

### 📜 Dokumentacja
- Przygotowanie raportu końcowego
- Przygotowanie prezentacji
- Aktualizacja repozytorium na GitHubie/GitLabie
