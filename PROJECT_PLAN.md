# 🛠️ Plan projektu: Wprowadzenie techniczne

## 📌 Issue 1: Przygotowanie środowiska lokalnego
- [x] Task 1.1: Zainstalować Pythona w odpowiedniej wersji (3.10/3.11)
- [ ] Task 1.2: Utworzyć i aktywować wirtualne środowisko (`python -m venv .venv`)
- [ ] Task 1.3: Zainstalować podstawowe zależności (`pip install -r requirements.txt`)

**Odpowiedzialny:** Kacper, Gracjan


---

## 📌 Issue 2: Dokumentacja projektu
- [ ] Task 2.1: Utworzyć plik `README.md` z opisem i instrukcją uruchamiania
- [ ] Task 2.2: Dodać sekcję "Roadmap" do `README.md`

**Odpowiedzialny:** Kacper  


---

## 📌 Issue 3: Konfiguracja GitHub
- [ ] Task 3.1: Utworzyć i zatwierdzić `.gitignore`
- [ ] Task 3.2: Dodać plik z licencją (np. MIT License)
- [ ] Task 3.3: Utworzyć "Projects" na GitHubie i przenieść tam Issues

**Odpowiedzialny:** Gracjan

---

## 📌 Issue 4: Projektowanie architektury
- [ ] Task 4.1: Zaplanować strukturę katalogów (`src/`, `tests/`, `data/`)
- [ ] Task 4.2: Utworzyć szkielety katalogów i plików (np. puste `__init__.py`)

**Odpowiedzialny:** Kacper  


---

## 📌 Issue 5: Testowe zadanie programistyczne
- [ ] Task 5.1: Utworzyć prosty skrypt `hello.py` i sprawdzić działanie środowiska

**Odpowiedzialny:** Gracjan


---

## 📌 Issue 6: Wybór głównych technologii i bibliotek
- [ ] Task 6.1: Zainstalować Python 3.x
- [ ] Task 6.2: Zainstalować Pandas, Numpy, Scikit-learn, XGBoost, TensorFlow, Keras, statsmodels
- [ ] Task 6.3: Skonfigurować środowisko wirtualne (np. venv)
- [ ] Task 6.4: Utworzyć plik `requirements.txt`

**Odpowiedzialny:** Kacper


---

## 📌 Issue 7: Zebranie i wstępna obróbka danych
- [ ] Task 7.1: Pozyskać dane historyczne rynku walutowego (np. EUR/USD) za lata 2015–2020
- [ ] Task 7.2: Wykonać eksploracyjną analizę danych (EDA)
- [ ] Task 7.3: Obróbka danych (przekształcenie na szereg czasowy, normalizacja)
- [ ] Task 7.4: Podzielić dane na zbiór treningowy i testowy

**Odpowiedzialny:** Kacper


---

## 📌 Issue 8: Implementacja i trenowanie modeli
### 3.1 Model ARIMA
- [ ] Task 8.1: Dobór parametrów (p, d, q) przy pomocy AIC/BIC
- [ ] Task 8.2: Wytrenowanie modelu na danych 2015–2020
- [ ] Task 8.3: Generowanie prognozy na rok 2021

### 3.2 Model XGBoost
- [ ] Task 8.4: Przygotowanie cech wejściowych (zmienne opóźnione)
- [ ] Task 8.5: Wytrenowanie modelu
- [ ] Task 8.6: Predykcja na rok 2021

### 3.3 Model LSTM
- [ ] Task 8.7: Przygotowanie sekwencji wejściowych (X, y)
- [ ] Task 8.8: Budowa sieci neuronowej (1-2 warstwy LSTM + Dense)
- [ ] Task 8.9: Wytrenowanie modelu
- [ ] Task 8.10: Predykcja na rok 2021

**Odpowiedzialny:** Kacper


---

## 📌 Issue 9: Ocena i porównanie modeli
- [ ] Task 9.1: Wygenerowanie prognoz każdego modelu na dane testowe
- [ ] Task 9.2: Obliczenie metryk błędu (RMSE, MAPE)
- [ ] Task 9.3: Pomiar czasu działania
- [ ] Task 9.4: Ocena interpretowalności modeli

**Odpowiedzialny:** Kacper


---

## 📌 Issue 10: Wyciągnięcie wniosków i rekomendacji
- [ ] Task 10.1: Analiza wyników i ocena najlepszych modeli
- [ ] Task 10.2: Zidentyfikowanie potencjalnych słabości
- [ ] Task 10.3: Sformułowanie rekomendacji do przyszłych prac

**Odpowiedzialny:** Kacper


---

## 📌 Issue 11: Opracowanie dokumentacji projektu
- [ ] Task 11.1: Przygotowanie raportu końcowego
- [ ] Task 11.2: Przygotowanie prezentacji
- [ ] Task 11.3: Aktualizacja repozytorium

**Odpowiedzialny:** Kacper


---

## 📌 Issue 12: Kamienie milowe
### KM1: Przygotowanie środowiska + zebranie danych
- **Termin:** 2–3 dni

### KM2: Wytrenowanie wszystkich modeli na 2015–2020
- **Termin:** 5–7 dni

### KM3: Wygenerowanie prognoz + ocena skuteczności
- **Termin:** 5 dni

### KM4: Wnioski, interpretacja, rekomendacje
- **Termin:** 3–4 dni

### KM5: Przygotowanie raportu, prezentacji i wrzucenie wszystkiego na GitHub/GitLab
- **Termin:** 2–3 dni

**Odpowiedzialny:** Kacper
