import pandas as pd
from sklearn.preprocessing import StandardScaler

# Załaduj dane
df = pd.read_csv('C:/Users/Kacper/Desktop/EUR_USD Historical Data.csv', sep=',')
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

# Możesz zapisać przekształcone dane do pliku, jeśli chcesz
df.to_csv('C:/Users/Kacper/Desktop/EUR_USD_Normalized.csv')
print(df.describe())
