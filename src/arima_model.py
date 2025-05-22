import pandas as pd
import pmdarima as pm
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import numpy as np
import os
import warnings

# --- 1. Wyciszenie FutureWarning ---
warnings.filterwarnings("ignore", category=FutureWarning)

# --- 2. Konfiguracja ścieżki do danych ---
script_dir = os.path.dirname(__file__)
data_path = os.path.join(script_dir, '..', 'data', 'EUR_USD_2015-2021_data.csv')
results_dir = os.path.join(script_dir, '..', 'results')
# Pełna ścieżka do pliku wyjściowego (wykresu)
plot_output_path = os.path.join(results_dir, 'arimax_forecast.png')

# --- 3. Ładowanie i wstępne przetwarzanie danych ---
print("\n--- Rozpoczynam ładowanie i przetwarzanie danych ---")

try:
    df = pd.read_csv(data_path, sep=',')
    df.columns = df.columns.str.strip().str.replace('"', '')

    # --- Diagnostyka Step 1: Po wczytaniu CSV i czyszczeniu nazw kolumn ---
    print(f"\n--- Diagnostyka Step 1: Po wczytaniu CSV i czyszczeniu nazw kolumn ---")
    print(f"Kształt DataFrame: {df.shape}")
    print(f"Nagłówki kolumn: {df.columns.tolist()}")
    print(f"Liczba NaNów w każdej kolumnie (isnull.sum()):\n{df.isnull().sum()}")

    # Sprawdzenie, czy kolumny 'Date' i 'Price' istnieją
    if 'Date' not in df.columns or 'Price' not in df.columns:
        raise ValueError("Brak kolumn 'Date' lub 'Price' w pliku CSV. Sprawdź nagłówki.")

    # Usunięcie kolumny 'Vol.' jeśli zawiera same NaNy (lub jeśli jest niepotrzebna)
    # Z analizy wynika, że 'Vol.' ma 1566 NaNów, czyli wszystkie.
    if 'Vol.' in df.columns:
        if df['Vol.'].isnull().all(): # Sprawdź, czy wszystkie wartości to NaN
            print("Kolumna 'Vol.' zawiera same wartości NaN i zostanie usunięta.")
            df.drop(columns=['Vol.'], inplace=True)
        # else: można by tu obsłużyć, jeśli 'Vol.' ma wartości, ale ma też NaNy,
        # np. wypełniając je zerami: df['Vol.'] = df['Vol.'].fillna(0)

    # Konwersja kolumny 'Price' na numeryczną (errors='coerce' zamieni błędy na NaN)
    df['Price'] = pd.to_numeric(df['Price'], errors='coerce')

    # Konwersja kolumny 'Date' na format daty (errors='coerce' zamieni błędy na NaT)
    df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y', errors='coerce')

    # Ustawienie kolumny 'Date' jako indeks
    df.set_index('Date', inplace=True)
    df = df.sort_index() # Upewnij się, że indeks jest posortowany chronologicznie

    # --- Diagnostyka Step 2: Po konwersji typów danych i ustawieniu indeksu ---
    print(f"\n--- Diagnostyka Step 2: Po konwersji typów danych i ustawieniu indeksu ---")
    print(f"Kształt DataFrame: {df.shape}")
    print(f"Typ indeksu: {df.index.dtype}")
    print(f"Liczba NaT w indeksie Date: {df.index.isnull().sum()}")
    print(f"Liczba NaNów w kolumnie Price po konwersji: {df['Price'].isnull().sum()}")
    print(f"Pierwsze 5 wierszy (head):\n{df.head().to_string()}")

    # Usuń wiersze, gdzie indeks (Date) jest NaT, lub gdzie 'Price' jest NaN
    # Robimy to na konkretnych kolumnach, żeby nie usuwać wszystkiego przez 'Vol.'
    df.dropna(subset=['Price'], inplace=True)
    df = df[df.index.notnull()] # Upewnij się, że indeks nie zawiera NaT

    # --- Diagnostyka Step 3: Po usunięciu NaNów/NaT z Price i Date ---
    print(f"\n--- Diagnostyka Step 3: Po usunięciu NaNów/NaT z Price i Date ---")
    print(f"Kształt DataFrame: {df.shape}")
    if df.empty:
        raise ValueError("DataFrame jest pusty po usunięciu błędnych dat lub cen. Sprawdź dane źródłowe.")

    # Obliczanie średnich kroczących (zmiennych egzogenicznych)
    df['SMA_7'] = df['Price'].rolling(window=7).mean()
    df['EMA_30'] = df['Price'].ewm(span=30, adjust=False).mean()

    # Ostateczne usuwanie NaNów, ale teraz TYLKO z kolumn, które mają sens
    # np. z kolumn SMA_7 i EMA_30, które będą miały NaNy na początku.
    # Ważne: df.dropna() bez argumentów usuwa wiersz, jeśli gdziekolwiek jest NaN.
    # Teraz, gdy Vol. zostało usunięte, dropna() usunie tylko początkowe wiersze.
    df.dropna(inplace=True)

    # --- Diagnostyka Step 4: Po obliczeniu średnich kroczących i ostatecznym dropna() ---
    print(f"\n--- Diagnostyka Step 4: Po obliczeniu średnich kroczących i ostatecznym dropna() ---")
    print(f"Kształt DataFrame: {df.shape}")
    print(f"Liczba NaNów w SMA_7: {df['SMA_7'].isnull().sum()}")
    print(f"Liczba NaNów w EMA_30: {df['EMA_30'].isnull().sum()}")
    print(f"Zakres dat w DataFrame po dropna(): {df.index.min()} do {df.index.max()}")
    print(f"Pierwsze 5 wierszy (head):\n{df.head().to_string()}")

    if df.empty:
        raise ValueError("DataFrame jest pusty po obliczeniu średnich kroczących i usunięciu NaNów. Upewnij się, że plik CSV zawiera wystarczająco dużo danych historycznych.")

except Exception as e:
    print(f"Wystąpił błąd podczas ładowania lub przetwarzania danych: {e}")
    print("Upewnij się, że plik 'EUR_USD Historical Data.csv' jest poprawnie sformatowany i zawiera dane.")
    exit() # Zakończ skrypt w przypadku błędu ładowania danych

# --- 4. Podział danych na zbiór treningowy i przygotowanie egzogenicznych ---
# Zdefiniuj datę początkową zbioru treningowego dynamicznie na podstawie dostępnych danych.
start_train_date = df.index.min()
end_train_date = '2020-12-31' # Koniec danych treningowych

# Podział danych na zbiór treningowy
train_data = df.loc[start_train_date:end_train_date, 'Price']
train_exog = df.loc[start_train_date:end_train_date, ['SMA_7', 'EMA_30']]

# Dodatkowa diagnostyka - sprawdź, czy train_data i train_exog są puste po loc[]
if train_data.empty or train_exog.empty:
    print(f"\nBŁĄD: Zbiór treningowy lub egzogeniczny jest pusty po wyborze dat.")
    print(f"Zakres dat po dropna(): {df.index.min()} do {df.index.max()}")
    print(f"Próbowano wybrać zakres: {start_train_date} do {end_train_date}")
    print(f"Liczba elementów w train_data: {len(train_data)}")
    print(f"Liczba elementów w train_exog: {len(train_exog)}")
    exit()

test_data = pd.Series()
n_periods = 365 # Liczba dni do prognozowania

print(f"\nLiczba punktów danych w zbiorze treningowym ({start_train_date.strftime('%Y-%m-%d')} do {end_train_date}): {len(train_data)}")
print(f"Liczba zmiennych egzogenicznych w zbiorze treningowym: {train_exog.shape[1]}")
print(f"Brak danych rzeczywistych na rok 2021. Prognoza zostanie wygenerowana dla {n_periods} dni od ostatniej daty w zbiorze treningowym.")

# --- 5. Model ARIMAX i prognozowanie ---
print("\n--- 5. Model ARIMAX z średnimi kroczącymi ---")

model_arima = pm.auto_arima(train_data,
                           exog=train_exog,
                           start_p=1, start_q=1,
                           test='adf',
                           max_p=5, max_q=5,
                           m=1,
                           d=None,
                           seasonal=False,
                           trace=True,
                           error_action='ignore',
                           suppress_warnings=True,
                           stepwise=True)

print(f"\nNajlepsze parametry ARIMAX (p, d, q) według auto_arima: {model_arima.order}")
print(f"Wartość AIC dla najlepszego modelu: {model_arima.aic()}")

last_train_exog = train_exog.iloc[-1]
last_train_date = train_data.index[-1]
forecast_exog_index = pd.date_range(start=last_train_date + pd.Timedelta(days=1), periods=n_periods, freq='D')

forecast_exog = pd.DataFrame(
    np.tile(last_train_exog.values, (n_periods, 1)),
    columns=train_exog.columns,
    index=forecast_exog_index
)

forecast, conf_int = model_arima.predict(n_periods=n_periods, X=forecast_exog, return_conf_int=True)

forecast_series = pd.Series(forecast, index=forecast_exog_index)
conf_int_df = pd.DataFrame(conf_int, index=forecast_exog_index, columns=['lower', 'upper'])

print("\nPrognoza na kolejny rok (pierwsze 5 dni):")
print(forecast_series.head())

# --- 6. Wizualizacja wyników ---
print("\n--- 6. Generowanie wykresu ---")
plt.figure(figsize=(14, 7))
plt.plot(train_data.index, train_data, label='Dane treningowe (2015-2020)', color='blue')
plt.plot(forecast_series.index, forecast_series, label='Prognoza ARIMAX (rok po danych treningowych)', color='red', linestyle='--')
plt.fill_between(conf_int_df.index, conf_int_df['lower'], conf_int_df['upper'], color='pink', alpha=0.3, label='Przedział ufności')
plt.title('Prognoza rynku walutowego EUR/USD modelem ARIMAX ze średnimi kroczącymi')
plt.xlabel('Data')
plt.ylabel('Cena')
plt.legend()
plt.grid(True)
plt.tight_layout()

if not os.path.exists(results_dir):
    os.makedirs(results_dir)

plt.savefig(plot_output_path) # Używamy już zdefiniowanej, pełnej ścieżki
print(f"Wykres został zapisany jako '{plot_output_path}'")

try:
    plt.show()
except Exception as e:
    print(f"Nie można wyświetlić wykresu interaktywnie: {e}")
    print(f"Sprawdź konfigurację backendu Matplotlib lub otwórz '{plot_output_path}' ręcznie.")

# --- 7. Ocena prognozy ---
print("\nBrak danych rzeczywistych na rok 2021 do oceny RMSE.")