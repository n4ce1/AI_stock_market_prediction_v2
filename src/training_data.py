import pandas as pd
import matplotlib.pyplot as plt
import os

data_path = os.path.abspath(os.path.join(os.path.dirname(file), "..", "data", "EUR_USD Historical Data.csv"))

df = pd.read_csv(data_path)
df.columns = df.columns.str.strip().str.replace('"', '')
df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')
df.set_index('Date', inplace=True)
df = df.sort_index()

df_filtered = df.loc['2015':'2020']
train = df_filtered.loc['2015':'2019']
test = df_filtered.loc['2020']

print(f"Rozmiar zbioru treningowego: {train.shape}")
print(f"Rozmiar zbioru testowego: {test.shape}")

plt.figure(figsize=(12, 6))
plt.plot(train.index, train['Price'], label='Train (2015–2019)', color='blue')
plt.plot(test.index, test['Price'], label='Test (2020)', color='orange')
plt.title('Podział danych EUR/USD (2015–2020)')
plt.xlabel('Data')
plt.ylabel('Cena')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()