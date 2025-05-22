import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os # Importuj moduł os do operacji na ścieżkach

# --- Konfiguracja ścieżki do danych ---
# dynamiczne określenie katalogu skryptu
script_dir = os.path.dirname(__file__)
# Ścieżka do pliku EUR_USD Historical Data.csv, który znajduje się w katalogu nadrzędnym
# względem katalogu src, gdzie ten skrypt się znajduje.
data_file_path = os.path.join(script_dir, '..', 'data', 'EUR_USD_2015-2021_data.csv')


# Załaduj dane
# ZMIANA: Używamy data_file_path
df = pd.read_csv(data_file_path, sep=',')
df.columns = df.columns.str.strip().str.replace('"', '')  # Oczyszczanie nazw kolumn

# Krok 1: Konwersja kolumny 'Date' na typ daty
# Użyj errors='coerce' dla większej odporności na błędy w danych
df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y', errors='coerce')

# Usuń wiersze, w których konwersja daty się nie powiodła
initial_rows = len(df)
df.dropna(subset=['Date'], inplace=True)
if len(df) < initial_rows:
    print(f"Usunięto {initial_rows - len(df)} wierszy z nieprawidłowymi datami.")

# Krok 2: Statystyki opisowe
print("\nStatystyki opisowe:")
print(df.describe())

# Krok 3: Wykresy
plt.figure(figsize=(14, 7))

# Wykres ceny
plt.subplot(2, 2, 1)
plt.plot(df['Date'], df['Price'], label='Cena', color='blue')
plt.title('Zmiana ceny EUR/USD w czasie')
plt.xlabel('Data')
plt.ylabel('Cena')
plt.grid(True)

# Wykres otwarcia (Open)
plt.subplot(2, 2, 2)
plt.plot(df['Date'], df['Open'], label='Open', color='orange')
plt.title('Cena otwarcia EUR/USD w czasie')
plt.xlabel('Data')
plt.ylabel('Cena')
plt.grid(True)

# Wykres zmiany procentowej
plt.subplot(2, 2, 3)
plt.plot(df['Date'], df['Change %'], label='Change %', color='green')
plt.title('Zmiana procentowa EUR/USD w czasie')
plt.xlabel('Data')
plt.ylabel('Zmiana (%)')
plt.grid(True)

# Rozkład ceny
plt.subplot(2, 2, 4)
sns.histplot(df['Price'], kde=True, color='blue')
plt.title('Rozkład ceny EUR/USD')
plt.xlabel('Cena')
plt.ylabel('Liczba wystąpień')

plt.tight_layout()
plt.show()