import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import os
import warnings

# --- 1. Wyciszenie FutureWarning ---
warnings.filterwarnings("ignore", category=FutureWarning)

# --- 2. Konfiguracja ścieżki do danych ---
script_dir = os.path.dirname(__file__)
data_path = os.path.join(script_dir, '..', 'data', 'EUR_USD_2015-2021_data.csv')

# --- 3. Ładowanie i wstępne przetwarzanie danych (powtórka z arima_model.py) ---
print("\n--- Rozpoczynam ładowanie i przetwarzanie danych dla XGBoost ---")

try:
    df = pd.read_csv(data_path, sep=',')
    df.columns = df.columns.str.strip().str.replace('"', '')

    if 'Date' not in df.columns or 'Price' not in df.columns:
        raise ValueError("Brak kolumn 'Date' lub 'Price' w pliku CSV. Sprawdź nagłówki.")

    if 'Vol.' in df.columns:
        if df['Vol.'].isnull().all():
            print("Kolumna 'Vol.' zawiera same wartości NaN i zostanie usunięta.")
            df.drop(columns=['Vol.'], inplace=True)

    df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
    df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y', errors='coerce')
    df.set_index('Date', inplace=True)
    df = df.sort_index()

    df.dropna(subset=['Price'], inplace=True)
    df = df[df.index.notnull()]

    if df.empty:
        raise ValueError("DataFrame jest pusty po usunięciu błędnych dat lub cen. Sprawdź dane źródłowe.")

    # Obliczanie średnich kroczących (zmiennych egzogenicznych)
    df['SMA_7'] = df['Price'].rolling(window=7).mean()
    df['EMA_30'] = df['Price'].ewm(span=30, adjust=False).mean()

    # Ostateczne usunięcie NaNów (głównie z początkowych średnich kroczących)
    df.dropna(inplace=True)

    if df.empty:
        raise ValueError("DataFrame jest pusty po obliczeniu średnich kroczących i usunięciu NaNów. Upewnij się, że plik CSV zawiera wystarczająco dużo danych historycznych.")

    print(f"Dane załadowane i przetworzone. Kształt DataFrame: {df.shape}")
    print(f"Zakres dat po czyszczeniu: {df.index.min()} do {df.index.max()}")

except Exception as e:
    print(f"Wystąpił błąd podczas ładowania lub przetwarzania danych: {e}")
    print("Upewnij się, że plik 'EUR_USD_2015-2021_data(normalized).csv' jest poprawnie sformatowany i zawiera dane.")
    exit()

# --- 4. Task 8.4: Przygotowanie cech wejściowych (zmienne opóźnione) ---
print("\n--- 4. Przygotowanie cech opóźnionych dla XGBoost ---")

# Definiowanie liczby opóźnień (lagów)
# Zwykle warto eksperymentować z tą wartością. Tutaj np. 5-7 dni roboczych.
n_lags = 7 # Użyjemy cen i średnich z ostatnich 7 dni roboczych

# Lista kolumn, dla których chcemy stworzyć opóźnienia
lag_cols = ['Price', 'SMA_7', 'EMA_30']

# Tworzenie cech opóźnionych
for col in lag_cols:
    for i in range(1, n_lags + 1):
        df[f'{col}_lag_{i}'] = df[col].shift(i)

# Usunięcie wierszy z NaNami powstałymi na skutek opóźnień
# Pierwsze 'n_lags' wierszy będzie miało NaN, więc je usuwamy
df_model = df.dropna()

# Definiowanie zmiennych niezależnych (X) i zmiennej zależnej (y)
# X to wszystkie stworzone cechy opóźnione oraz aktualne wartości SMA_7 i EMA_30
# y to 'Price' (cena, którą chcemy prognozować)
features = [f'{col}_lag_{i}' for col in lag_cols for i in range(1, n_lags + 1)]

# Dodatkowo, jeśli chcemy użyć aktualnych SMA_7 i EMA_30 jako cech,
# ale pamiętajmy, że do prognozowania w przyszłość będziemy musieli je też prognozować.
# Na razie trzymamy się tylko cech opóźnionych, żeby było prostsze prognozowanie przyszłości.
# Jeśli chcesz dodać np. aktualny miesiąc, dzień tygodnia jako cechy:
df_model['day_of_week'] = df_model.index.dayofweek # Poniedziałek=0, Niedziela=6
df_model['day_of_year'] = df_model.index.dayofyear
df_model['month'] = df_model.index.month

features.extend(['day_of_week', 'day_of_year', 'month'])


X = df_model[features]
y = df_model['Price']

print(f"Kształt danych po przygotowaniu cech opóźnionych: X={X.shape}, y={y.shape}")
print(f"Pierwsze 5 wierszy X (head):\n{X.head().to_string()}")


# --- 5. Task 8.5: Wytrenowanie modelu XGBoost ---
print("\n--- 5. Trenowanie modelu XGBoost ---")

# Podział danych na zbiór treningowy (do końca 2020) i testowy (nie używamy, ale dla spójności)
# Zamiast train_test_split, użyjemy podziału czasowego.
# Dane treningowe: od początku dostępnych danych do końca 2020 roku
# Dane testowe (dla oceny modelu na historycznych danych): początek 2021 roku (jeśli dostępne)
# W tym przypadku, dane treningowe to X i y, które już są przygotowane.
# Będziemy prognozować przyszłość na podstawie danych treningowych.

X_train = X.loc[X.index.min():'2020-12-31']
y_train = y.loc[y.index.min():'2020-12-31']

# Sprawdzenie, czy zbiory treningowe nie są puste
if X_train.empty or y_train.empty:
    raise ValueError("Zbiór treningowy X_train lub y_train jest pusty. Sprawdź zakres dat.")

print(f"Kształt zbioru treningowego: X_train={X_train.shape}, y_train={y_train.shape}")
print(f"Zakres dat zbioru treningowego: {X_train.index.min()} do {X_train.index.max()}")


# Inicjalizacja i trenowanie modelu XGBoost Regressor
# Możesz dostosować parametry (np. n_estimators, learning_rate, max_depth) dla lepszych wyników.
# Domyślne parametry są często dobrym punktem wyjścia.
xgb_model = xgb.XGBRegressor(
    objective='reg:squarederror', # Cel regresji
    n_estimators=1000,           # Liczba drzew decyzyjnych
    learning_rate=0.05,          # Współczynnik uczenia
    max_depth=5,                 # Maksymalna głębokość drzewa
    subsample=0.7,               # Proporcja próbek do trenowania każdego drzewa
    colsample_bytree=0.7,        # Proporcja cech do trenowania każdego drzewa
    random_state=42,             # Dla powtarzalności wyników
    n_jobs=-1                    # Użyj wszystkich dostępnych rdzeni procesora
)

xgb_model.fit(X_train, y_train)

print("Model XGBoost został wytrenowany.")

# --- 6. Task 8.6: Predykcja na rok 2021 ---
print("\n--- 6. Predykcja na rok 2021 (dni robocze) ---")

# Generowanie przyszłych dat (tylko dni robocze)
last_train_date = y_train.index[-1]
future_dates = pd.bdate_range(start=last_train_date + pd.Timedelta(days=1), end='2021-12-31')
n_periods_forecast = len(future_dates)

# Aby prognozować przyszłość, potrzebujemy "przyszłych" wartości cech opóźnionych.
# To jest kluczowy i często trudny aspekt prognozowania szeregów czasowych z cechami.
# Stosujemy podejście iteracyjne (step-by-step forecasting), gdzie prognoza z jednego dnia
# staje się cechą opóźnioną dla następnego dnia.

forecast_prices = []
current_X = X_train.iloc[-1].to_dict() # Pobierz ostatnie znane cechy z treningu

# Iteracyjna prognoza dla każdego dnia roboczego w 2021 roku
for i, forecast_date in enumerate(future_dates):
    # Utwórz DataFrame dla jednej prognozy
    X_single_forecast = pd.DataFrame([current_X], index=[forecast_date])

    # Dodaj cechy daty dla pojedynczej prognozy
    X_single_forecast['day_of_week'] = forecast_date.dayofweek
    X_single_forecast['day_of_year'] = forecast_date.dayofyear
    X_single_forecast['month'] = forecast_date.month

    # Zapewnij, że kolumny są w tej samej kolejności co podczas trenowania
    X_single_forecast = X_single_forecast[features]

    # Wykonaj predykcję
    predicted_price = xgb_model.predict(X_single_forecast)[0]
    forecast_prices.append(predicted_price)

    # Aktualizuj 'current_X' dla następnej iteracji (przesuń opóźnienia)
    # Price_lag_1 staje się Price_lag_2, Price_lag_2 staje się Price_lag_3 itd.
    # Nowy Price_lag_1 to właśnie prognozowana cena
    for lag in range(n_lags, 1, -1):
        for col_name in lag_cols:
            current_X[f'{col_name}_lag_{lag}'] = current_X[f'{col_name}_lag_{lag-1}']

    # Ustaw nową prognozowaną cenę jako Price_lag_1 dla następnego kroku
    current_X['Price_lag_1'] = predicted_price

    # Trzeba też zaktualizować średnie kroczące dla następnego kroku.
    # To jest najtrudniejsze, bo SMA/EMA zależą od przyszłych (nieznanych) cen.
    # Najprostsze podejście: po prostu używamy prognozowanej ceny jako 'Price'
    # i przeliczamy SMA/EMA w uproszczony sposób.
    # DLA CELÓW TEGO ĆWICZENIA PRZYJMUJEMY UPROSZCZENIE:
    # ŻE PROGNOZOWANE ŚREDNIE KROCZĄCE SĄ RÓWNE OSTATNIEJ ZNANEJ WARTOŚCI.
    # To jest podobne do tego, co zrobiliśmy w ARIMA.
    # Prawidłowo wymagałoby to modelowania średnich kroczących osobno lub w bardziej złożony sposób.
    # Na razie zostawimy aktualne wartości SMA/EMA_lag_1 jako ostatnie znane wartości,
    # co oznacza, że one nie będą się zmieniać dynamicznie w prognozie, tylko cena.
    # Jeśli chcesz, żeby średnie były aktualizowane, musisz zastosować bardziej skomplikowaną logikę.
    # Na razie kontynuujemy z 'Price_lag_1' jako jedyną dynamiczną cechą opóźnioną.

    # Przykład: Zaktualizuj EMA_30_lag_1 (najbardziej dynamiczne z lag_cols po Price)
    # To jest bardzo uproszczone i nie odzwierciedla prawdziwych średnich kroczących.
    # Można by zastosować: current_X['EMA_30_lag_1'] = (predicted_price * (2/31)) + (current_X['EMA_30_lag_1'] * (1 - (2/31)))
    # ale to wymagałoby śledzenia całego okna EMA/SMA, co jest skomplikowane w tym iteracyjnym procesie.
    # NA POTRZEBY TEGO ZADANIA, AKCEPTUJEMY, ŻE TYLKO LAGOWANE CENY BĘDĄ SIĘ DYNAMICZNIE ZMIENIAĆ.
    # Średnie kroczące będą propagować swoje ostatnie znane wartości.

# Tworzenie serii Pandas dla prognozy
forecast_series = pd.Series(forecast_prices, index=future_dates)

print("\nPrognoza na rok 2021 (pierwsze 5 dni roboczych):")
print(forecast_series.head())

# --- 7. Wizualizacja wyników ---
print("\n--- 7. Generowanie wykresu ---")
plt.figure(figsize=(14, 7))
plt.plot(y_train.index, y_train, label='Dane treningowe (2015-2020)', color='blue')
plt.plot(forecast_series.index, forecast_series, label='Prognoza XGBoost (dni robocze 2021)', color='green', linestyle='--')
plt.title('Prognoza rynku walutowego EUR/USD modelem XGBoost z cechami opóźnionymi')
plt.xlabel('Data')
plt.ylabel('Cena')
plt.legend()
plt.grid(True)
plt.tight_layout()

output_dir_plot = os.path.join(script_dir, '..', 'results')
# Upewnij się, że katalog 'data' istnieje
if not os.path.exists(output_dir_plot):
    os.makedirs(output_dir_plot)

plot_file_name = 'xgboost_forecast_business_days.png'
plot_output_path = os.path.join(output_dir_plot, plot_file_name)

plt.savefig(plot_output_path)
print(f"Wykres został zapisany jako '{plot_output_path}'")

try:
    plt.show()
except Exception as e:
    print(f"Nie można wyświetlić wykresu interaktywnie: {e}")
    print(f"Sprawdź konfigurację backendu Matplotlib lub otwórz '{plot_output_path}' ręcznie.")

# --- 8. Ocena modelu (opcjonalnie, jeśli masz dane testowe) ---
# Jeśli miałbyś dane z 2021 roku (np. test_data), mógłbyś obliczyć RMSE:
# from sklearn.metrics import mean_squared_error
# import numpy as np
# rmse = np.sqrt(mean_squared_error(y_test, predictions_on_test_data))
# print(f"RMSE na danych testowych: {rmse}")

print("\nBrak danych rzeczywistych na rok 2021 do oceny RMSE.")