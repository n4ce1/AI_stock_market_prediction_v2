# 🛠️ Plan projektu: Wprowadzenie techniczne

## 📌 Issue 1: Przygotowanie środowiska lokalnego
- [x] Task 1.1: Zainstalować Python 3.12.3
- [x] Task 1.2: Zainstalować Pandas, Numpy, Scikit-learn, XGBoost, TensorFlow, Keras, statsmodels
- [x] Task 1.3: Skonfigurować środowisko wirtualne (np. venv)
- [x] Task 1.4: Utworzyć plik `requirements.txt`

**Odpowiedzialny:** Kacper, Gracjan


---

## 📌 Issue 2: Dokumentacja projektu
- [x] Task 2.1: Utworzyć plik `README.md` z opisem i instrukcją uruchamiania
- [x] Task 2.2: Dodać sekcję "Roadmap" do `README.md`

**Odpowiedzialny:** Kacper  


---

## 📌 Issue 3: Konfiguracja GitHub
- [x] Task 3.1: Utworzyć i zatwierdzić `.gitignore`
- [x] Task 3.2: Dodać plik z licencją (np. MIT License)
- [x] Task 3.3: Utworzyć "Projects" na GitHubie i przenieść tam Issues

**Odpowiedzialny:** Gracjan

---

## 📌 Issue 4: Projektowanie architektury
- [x] Task 4.1: Zaplanować strukturę katalogów (`src/`, `tests/`, `data/`)
- [x] Task 4.2: Utworzyć szkielety katalogów i plików (np. puste `__init__.py`)

**Odpowiedzialny:** Kacper  


---

## 📌 Issue 5: Testowe zadanie programistyczne
- [x] Task 5.1: Utworzyć prosty skrypt `hello.py` i sprawdzić działanie środowiska

**Odpowiedzialny:** Gracjan


---

## 📌 Issue 6: Zebranie i wstępna obróbka danych
- [x] Task 6.1: Pozyskać dane historyczne rynku walutowego (np. EUR/USD) za lata 2015–2020
- [x] Task 6.2: Wykonać eksploracyjną analizę danych (EDA)
- [x] Task 6.3: Obróbka danych (przekształcenie na szereg czasowy, normalizacja)
- [x] Task 6.4: Podzielić dane na zbiór treningowy i testowy

**Odpowiedzialny:** Kacper


---

## 📌 Issue 7: Implementacja i trenowanie modeli
### 3.1 Model ARIMA
- [x] Task 7.1: Dobór parametrów (p, d, q) przy pomocy AIC/BIC
- [x] Task 7.2: Wytrenowanie modelu na danych 2015–2020
- [x] Task 7.3: Generowanie prognozy na rok 2021

**Odpowiedzialny:** Kacper


### 3.2 Model XGBoost
- [x] Task 7.4: Przygotowanie cech wejściowych (zmienne opóźnione)
- [x] Task 7.5: Wytrenowanie modelu
- [x] Task 7.6: Predykcja na rok 2021

**Odpowiedzialny:** Gracjan


---

## 📌 Issue 8: Ocena i porównanie modeli
- [ ] Task 8.1: Wygenerowanie prognoz każdego modelu na dane testowe
- [ ] Task 8.2: Obliczenie metryk błędu (RMSE, MAPE)
- [ ] Task 8.3: Pomiar czasu działania
- [ ] Task 8.4: Ocena interpretowalności modeli

**Odpowiedzialny:** Kacper


---

## 📌 Issue 9: Wyciągnięcie wniosków i rekomendacji
- [ ] Task 9.1: Analiza wyników i ocena najlepszych modeli
- [ ] Task 9.2: Zidentyfikowanie potencjalnych słabości
- [ ] Task 9.3: Sformułowanie rekomendacji do przyszłych prac

**Odpowiedzialny:** Kacper


---

## 📌 Issue 10: Opracowanie dokumentacji projektu
- [ ] Task 10.1: Przygotowanie raportu końcowego
- [ ] Task 10.2: Przygotowanie prezentacji
- [ ] Task 10.3: Aktualizacja repozytorium

**Odpowiedzialny:** Kacper


---

## 📌 Issue 11: Kamienie milowe
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
