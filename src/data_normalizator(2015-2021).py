import pandas as pd
from sklearn.preprocessing import StandardScaler
import os # Importuj moduł os do operacji na ścieżkach

# --- Konfiguracja ścieżek ---
script_dir = os.path.dirname(__file__) # Katalog, w którym jest ten skrypt (np. src)
data_dir = os.path.join(script_dir, '..', 'data') # Katalog 'data' w głównym katalogu (rodzic src)

# Pełne ścieżki do plików wejściowego i wyjściowego
input_file_path = os.path.join(data_dir, 'EUR_USD_2015-2021_data.csv')
output_file_path = os.path.join(data_dir, 'EUR_USD_2015-2021_data(normalized).csv')


# Załaduj dane
# ZMIANA: Używamy input_file_path
df = pd.read_csv(input_file_path, sep=',')
df.columns = df.columns.str.strip().str.replace('"', '')  # Oczyszczanie nazw kolumn

# Konwersja daty na format datetime i ustawienie jej jako indeks
df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')
df.set_index('Date', inplace=True)

# Krok 1: Normalizacja danych (Price, Open, High, Low)
scaler = StandardScaler()
df[['Price', 'Open', 'High', 'Low']] = scaler.fit_transform(df[['Price', 'Open', 'High', 'Low']])

# Krok 2: Wyświetlenie przekształconych danych
print("\nPierwsze kilka wierszy po normalizacji:")
print(df.head(150))

# Krok 3: Sprawdzenie, czy dane są już przekształcone na szereg czasowy
print("\nIndeks danych po ustawieniu daty:")
print(df.index)

# Zapisz przekształcone dane do pliku w docelowym katalogu
# ZMIANA: Używamy output_file_path
df.to_csv(output_file_path)
print(df.describe())