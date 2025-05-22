import pandas as pd
import os # Dodajemy import os, bo jest potrzebny do operacji na ścieżkach

# --- Konfiguracja ścieżek ---
script_dir = os.path.dirname(__file__) # Katalog, w którym jest ten skrypt (np. src)
data_dir = os.path.join(script_dir, '..', 'data') # Katalog 'data' w głównym katalogu (rodzic src)

# Pełne ścieżki do plików wejściowego i wyjściowego
input_file_path = os.path.join(data_dir, 'EUR_USD_2021-2022_data.csv')
output_file_path = os.path.join(data_dir, 'EUR_USD_2021-2022_data(normalized).csv')

# Wczytanie danych z poprawnej ścieżki
# Używamy zmiennej input_file_path zamiast poszczególnych komponentów
df = pd.read_csv(input_file_path, sep=',', decimal=',', encoding='utf-8')

print("Nagłówki przed zmianą:", df.columns)
print("Liczba kolumn:", len(df.columns))

if len(df.columns) != 7:
    print("UWAGA: Plik nie ma 7 kolumn. Sprawdź separator lub format pliku!")
    pass

df.columns = ['Date', 'Price', 'Open', 'High', 'Low', 'Vol.', 'Change %']

# Zamień format daty z DD.MM.RRRR na MM/DD/RRRR
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True).dt.strftime('%m/%d/%Y')

# Zapis do pliku wyjściowego z poprawnej ścieżki
df.to_csv(output_file_path, sep=',', index=False, encoding='utf-8')
print("Plik został poprawnie przekształcony!")