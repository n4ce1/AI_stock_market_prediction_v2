import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import os
import warnings
import re # Do wyrażeń regularnych

# --- 1. Wyciszenie FutureWarning ---
warnings.filterwarnings("ignore", category=FutureWarning)

# --- FUNKCJA DO GENEROWANIA UNIKALNYCH NAZW PLIKÓW ---
def get_next_file_number(directory, base_name, extension):
    """
    Znajduje najwyższy numer w nazwach plików o danym wzorcu
    i zwraca kolejny numer do użycia.
    """
    max_num = 0
    # Wzorzec dopasowania np. 'AI_prediction_2021-2022_(\d+).csv' lub 'xgboost_prediction_(\d+).png'
    pattern = re.compile(rf"^{re.escape(base_name)}_(\d+){re.escape(extension)}$")
    
    # Upewnij się, że katalog istnieje
    if not os.path.exists(directory):
        os.makedirs(directory) # Tworzy katalog, jeśli nie istnieje

    for filename in os.listdir(directory):
        match = pattern.match(filename)
        if match:
            try:
                num = int(match.group(1))
                if num > max_num:
                    max_num = num
            except ValueError:
                # Ignoruj pliki, których numer nie jest poprawną liczbą
                pass
    return max_num + 1

# --- 2. Konfiguracja ścieżki do danych ---
script_dir = os.path.dirname(__file__)
data_historical_path = os.path.join(script_dir, '..', 'data', 'EUR_USD_2015-2021_data.csv')
# Zakładamy, że ten plik zawiera SUROWE, ale już sformatowane dane 2021-2022
data_2021_2022_raw_formatted_path = os.path.join(script_dir, '..', 'data', 'EUR_USD_2021-2022_data(normalized).csv')


# --- 3. Ładowanie i wstępne przetwarzanie danych HISTORYCZNYCH (2015-2020) ---
print("\n--- Rozpoczynam ładowanie i przetwarzanie danych historycznych (2015-2020) dla XGBoost ---")

try:
    df_historical = pd.read_csv(data_historical_path, sep=',')
    df_historical.columns = df_historical.columns.str.strip().str.replace('"', '')

    if 'Date' not in df_historical.columns or 'Price' not in df_historical.columns:
        raise ValueError("Brak kolumn 'Date' lub 'Price' w pliku historycznym CSV. Sprawdź nagłówki.")

    if 'Vol.' in df_historical.columns:
        if df_historical['Vol.'].isnull().all():
            print("Kolumna 'Vol.' zawiera same wartości NaN i zostanie usunięta.")
            df_historical.drop(columns=['Vol.'], inplace=True)

    df_historical['Price'] = pd.to_numeric(df_historical['Price'], errors='coerce')
    df_historical['Date'] = pd.to_datetime(df_historical['Date'], format='%m/%d/%Y', errors='coerce')
    df_historical.set_index('Date', inplace=True)
    df_historical = df_historical.sort_index()

    df_historical.dropna(subset=['Price'], inplace=True)
    df_historical = df_historical[df_historical.index.notnull()]

    if df_historical.empty:
        raise ValueError("DataFrame historyczny jest pusty po usunięciu błędnych dat lub cen. Sprawdź dane źródłowe.")

    # --- KLUCZOWE: Normalizacja danych historycznych (Price) ---
    scaler = StandardScaler()
    # Fitujemy scaler TYLKO na danych treningowych, aby zapobiec wyciekowi danych
    df_historical.loc[:, 'Price_scaled'] = scaler.fit_transform(df_historical[['Price']])

    # Obliczanie średnich kroczących (na SKALOWANYCH cenach)
    df_historical.loc[:, 'SMA_7_scaled'] = df_historical['Price_scaled'].rolling(window=7).mean()
    df_historical.loc[:, 'EMA_30_scaled'] = df_historical['Price_scaled'].ewm(span=30, adjust=False).mean()

    # Ostateczne usunięcie NaNów (głównie z początkowych średnich kroczących)
    df_historical.dropna(inplace=True)

    if df_historical.empty:
        raise ValueError("DataFrame historyczny jest pusty po obliczeniu średnich kroczących i usunięciu NaNów. Upewnij się, że plik CSV zawiera wystarczająco dużo danych historycznych.")

    print(f"Dane historyczne załadowane, przeskalowane i przetworzone. Kształt DataFrame: {df_historical.shape}")
    print(f"Zakres dat po czyszczeniu (historyczne): {df_historical.index.min()} do {df_historical.index.max()}")

except Exception as e:
    print(f"Wystąpił błąd podczas ładowania lub przetwarzania danych historycznych: {e}")
    print("Upewnij się, że plik 'EUR_USD_2015-2021_data(normalized).csv' jest poprawnie sformatowany i zawiera dane.")
    exit()

# --- 4. Przygotowanie cech wejściowych (zmienne opóźnione) dla danych historycznych ---
print("\n--- 4. Przygotowanie cech opóźnionych dla XGBoost (dane historyczne) ---")

n_lags = 7
# Używamy Price_scaled do tworzenia lagów, a nie oryginalnego Price
lag_cols = ['Price_scaled', 'SMA_7_scaled', 'EMA_30_scaled']

for col in lag_cols:
    for i in range(1, n_lags + 1):
        df_historical.loc[:, f'{col}_lag_{i}'] = df_historical[col].shift(i)

df_model_historical = df_historical.dropna().copy() # Dodano .copy() aby uniknąć SettingWithCopyWarning

df_model_historical.loc[:, 'day_of_week'] = df_model_historical.index.dayofweek
df_model_historical.loc[:, 'day_of_year'] = df_model_historical.index.dayofyear
df_model_historical.loc[:, 'month'] = df_model_historical.index.month

features = [f'{col}_lag_{i}' for col in lag_cols for i in range(1, n_lags + 1)]
features.extend(['day_of_week', 'day_of_year', 'month'])

# Teraz zmienna zależna (y) to 'Price_scaled'
X_historical = df_model_historical[features]
y_historical = df_model_historical['Price_scaled'] # Cel prognozy to przeskalowana cena

print(f"Kształt danych po przygotowaniu cech opóźnionych (historyczne): X={X_historical.shape}, y={y_historical.shape}")

# --- 5. Wytrenowanie modelu XGBoost ---
print("\n--- 5. Trenowanie modelu XGBoost ---")

X_train = X_historical.loc[X_historical.index.min():'2020-12-31']
y_train = y_historical.loc[y_historical.index.min():'2020-12-31']

if X_train.empty or y_train.empty:
    raise ValueError("Zbiór treningowy X_train lub y_train jest pusty. Sprawdź zakres dat.")

print(f"Kształt zbioru treningowego: {X_train.shape}, y_train={y_train.shape}")
print(f"Zakres dat zbioru treningowego: {X_train.index.min()} do {X_train.index.max()}")

xgb_model = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.7,
    colsample_bytree=0.7,
    random_state=42,
    n_jobs=-1
)

xgb_model.fit(X_train, y_train)

print("Model XGBoost został wytrenowany.")


# --- 7. Ładowanie i SKALOWANIE DANYCH RZECZYWISTYCH (2021-2022) ---
print("\n--- 7. Ładowanie i SKALOWANIE danych rzeczywistych (2021-2022) ---")

try:
    # Wczytujemy plik, który zawiera SUROWE, ale już sformatowane dane
    df_actual_21_22_raw_formatted = pd.read_csv(data_2021_2022_raw_formatted_path, sep=',')
    df_actual_21_22_raw_formatted.columns = df_actual_21_22_raw_formatted.columns.str.strip().str.replace('"', '')

    # Upewniamy się, że kolumna 'Date' jest indeksem typu datetime
    if 'Date' in df_actual_21_22_raw_formatted.columns:
        df_actual_21_22_raw_formatted.loc[:, 'Date'] = pd.to_datetime(df_actual_21_22_raw_formatted['Date'])
        df_actual_21_22_raw_formatted.set_index('Date', inplace=True)
    elif df_actual_21_22_raw_formatted.index.name != 'Date' or not pd.api.types.is_datetime64_any_dtype(df_actual_21_22_raw_formatted.index):
        # Fallback jeśli 'Date' nie jest kolumną, ale indeks nie jest datetime
        df_actual_21_22_raw_formatted.index = pd.to_datetime(df_actual_21_22_raw_formatted.index, errors='coerce')

    df_actual_21_22_raw_formatted = df_actual_21_22_raw_formatted.sort_index()
    df_actual_21_22_raw_formatted.dropna(subset=['Price'], inplace=True)

    if df_actual_21_22_raw_formatted.empty:
        raise ValueError("DataFrame rzeczywistych danych 2021-2022 jest pusty. Sprawdź plik.")

    # --- KLUCZOWE: SKALOWANIE DANYCH RZECZYWISTYCH PRZY UŻYCIU TEGO SAMEGO SCALERA! ---
    # Używamy scaler.transform(), NIE scaler.fit_transform()
    actual_prices_21_22_scaled = scaler.transform(df_actual_21_22_raw_formatted[['Price']])
    actual_prices_21_22_scaled_series = pd.Series(actual_prices_21_22_scaled.flatten(), index=df_actual_21_22_raw_formatted.index)

    print(f"Rzeczywiste dane 2021-2022 załadowane i przeskalowane. Kształt: {actual_prices_21_22_scaled_series.shape}")
    print(f"Zakres dat (rzeczywiste 2021-2022 przeskalowane): {actual_prices_21_22_scaled_series.index.min()} do {actual_prices_21_22_scaled_series.index.max()}")

except Exception as e:
    print(f"Wystąpił błąd podczas ładowania lub przetwarzania danych 2021-2022: {e}")
    print("Upewnij się, że plik 'EUR_USD_2021-2022_data(normalized).csv.csv' jest poprawnie sformatowany i zawiera dane.")
    exit()

# --- 6. Predykcja na lata 2021-2022 (dni robocze) ---
print("\n--- 6. Predykcja na lata 2021-2022 (dni robocze) ---")

last_train_date = y_train.index[-1]
# Ustawiamy zakres prognozy tak, aby pokrywał się z dostępnymi danymi rzeczywistymi
# Prognozujemy od dnia po ostatnim treningowym do ostatniego dnia w DANYCH RZECZYWISTYCH
future_dates = pd.bdate_range(start=last_train_date + pd.Timedelta(days=1), end=actual_prices_21_22_scaled_series.index.max())

forecast_prices_scaled = [] # Będziemy prognozować skalowane ceny
current_X = X_train.iloc[-1].to_dict()

for i, forecast_date in enumerate(future_dates):
    X_single_forecast = pd.DataFrame([current_X], index=[forecast_date])

    X_single_forecast.loc[:, 'day_of_week'] = forecast_date.dayofweek
    X_single_forecast.loc[:, 'day_of_year'] = forecast_date.dayofyear
    X_single_forecast.loc[:, 'month'] = forecast_date.month

    # Zapewnij, że kolumny są w tej samej kolejności co podczas trenowania
    X_single_forecast = X_single_forecast[features]

    predicted_price_scaled = xgb_model.predict(X_single_forecast)[0]
    forecast_prices_scaled.append(predicted_price_scaled)

    # Aktualizuj 'current_X' dla następnej iteracji (przesuń opóźnienia)
    for lag in range(n_lags, 1, -1):
        for col_name in lag_cols:
            if f'{col_name}_lag_{lag-1}' in current_X:
                current_X[f'{col_name}_lag_{lag}'] = current_X[f'{col_name}_lag_{lag-1}']
            else:
                current_X[f'{col_name}_lag_{lag}'] = 0.0

    current_X['Price_scaled_lag_1'] = predicted_price_scaled # Ustawiamy prognozowaną SKALOWANĄ cenę

forecast_series_scaled = pd.Series(forecast_prices_scaled, index=future_dates)

print("\nPrognoza na lata 2021-2022 (pierwsze 5 dni roboczych - skalowane):")
print(forecast_series_scaled.head())
print("\nPrognoza na lata 2021-2022 (ostatnie 5 dni roboczych - skalowane):")
print(forecast_series_scaled.tail())


# --- 8. Porównanie i ocena wyników ---
print("\n--- 8. Porównanie prognozy z rzeczywistymi danymi (2021-2022) ---")

common_dates = forecast_series_scaled.index.intersection(actual_prices_21_22_scaled_series.index)

if common_dates.empty:
    print("Brak wspólnych dat między prognozą a danymi rzeczywistymi. Nie można porównać.")
    print("Sprawdź zakresy dat prognozy i pliku rzeczywistego.")
    y_true = pd.Series([])
    y_pred = pd.Series([])
else:
    y_true = actual_prices_21_22_scaled_series.loc[common_dates]
    y_pred = forecast_series_scaled.loc[common_dates]

    if len(y_true) > 0 and len(y_pred) > 0:
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)

        print(f"RMSE (Root Mean Squared Error) dla 2021-2022 (skalowane): {rmse:.4f}")
        print(f"MAE (Mean Absolute Error) dla 2021-2022 (skalowane): {mae:.4f}")
    else:
        print("Brak wystarczających danych do obliczenia metryk (RMSE/MAE).")

# --- 9. Wizualizacja wyników z porównaniem ---
print("\n--- 9. Generowanie wykresu z porównaniem prognozy i danych rzeczywistych ---")
plt.figure(figsize=(16, 8))

# Dane treningowe (przeskalowane)
plt.plot(y_train.index, y_train, label='Dane treningowe (2015-2020) - Skalowane', color='blue', alpha=0.7)

# Prognoza XGBoost (skalowane)
plt.plot(forecast_series_scaled.index, forecast_series_scaled, label='Prognoza XGBoost (2021-2022) - Skalowane', color='green', linestyle='--', linewidth=2)

# Rzeczywiste dane 2021-2022 (skalowane)
if not y_true.empty:
    plt.plot(y_true.index, y_true, label='Rzeczywiste dane (2021-2022) - Skalowane', color='red', linestyle='-', linewidth=1.5, alpha=0.8)

plt.title('Prognoza rynku walutowego EUR/USD modelem XGBoost vs. Rzeczywiste dane (Skala znormalizowana)')
plt.xlabel('Data')
plt.ylabel('Znormalizowana cena')
plt.legend()
plt.grid(True)
plt.tight_layout()

# --- Zapis wykresu z unikalną nazwą ---
plot_base_name = 'xgboost_forecast_vs_actuals_2021_2022_normalized'
plot_extension = '.png'
output_dir = os.path.join(script_dir, '..', 'results') # Katalog 'data' w folderze nadrzędnym
plot_num = get_next_file_number(output_dir, plot_base_name, plot_extension)
plot_file_name = f"{plot_base_name}_{plot_num}{plot_extension}"
plot_output_path = os.path.join(output_dir, plot_file_name)

plt.savefig(plot_output_path)
print(f"Wykres z porównaniem został zapisany jako '{plot_file_name}'")

try:
    plt.show()
except Exception as e:
    print(f"Nie można wyświetlić wykresu interaktywnie: {e}")
    print(f"Sprawdź konfigurację backendu Matplotlib lub otwórz '{plot_file_name}' ręcznie.")

# --- Zapis prognozy na 2021-2022 do pliku CSV z unikalną nazwą (po odwróceniu skalowania) ---
csv_base_name = 'AI_prediction_2021-2022' # Nazwa bazowa pliku z prognozą
csv_extension = '.csv'
csv_num = get_next_file_number(output_dir, csv_base_name, csv_extension)
csv_file_name = f"{csv_base_name}_{csv_num}{csv_extension}"
csv_output_path = os.path.join(output_dir, csv_file_name)

# Odwrócenie normalizacji dla prognozy na 2021-2022, aby uzyskać rzeczywiste ceny
forecast_series_actual_21_22 = pd.Series(scaler.inverse_transform(forecast_series_scaled.values.reshape(-1, 1)).flatten(), index=forecast_series_scaled.index)


# Upewnij się, że DataFrame do zapisu ma odpowiednią kolumnę
df_forecast_to_save = pd.DataFrame({
    'Date': forecast_series_actual_21_22.index,
    'Predicted_Price_2021_2022': forecast_series_actual_21_22.values
})
df_forecast_to_save.to_csv(csv_output_path, index=False) # index=False, aby nie zapisywać indeksu jako kolumny
print(f"\nPrognoza na lata 2021-2022 (rzeczywiste wartości) została zapisana do pliku: {csv_file_name}")

print("\n--- Zakończono analizę XGBoost z porównaniem prognozy ---")