import pandas as pd
import pmdarima as pm
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import numpy as np

# Załaduj i przygotuj dane (poniższy fragment zakłada, że masz już zdefiniowane 'train' i 'test')
# Dla celów demonstracyjnych, załaduję dane podobnie jak w Twoich skryptach
# Pamiętaj, aby dostosować ścieżkę do pliku danych
data_path = 'C:/Users/Kacper/Desktop/EUR_USD Historical Data.csv' # Zastąp prawidłową ścieżką!
df = pd.read_csv(data_path, sep=',')
df.columns = df.columns.str.strip().str.replace('"', '')
df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')
df.set_index('Date', inplace=True)
df = df.sort_index()

# Podział danych na zbiór treningowy (2015-2020) i testowy (2021)
# Zgodnie z Twoim planem, trenujemy na 2015-2020 i prognozujemy 2021
train_data = df.loc['2015':'2020', 'Price']
test_data = df.loc['2021', 'Price'] # Dane testowe na rok 2021

# Sprawdzenie, czy dane testowe na 2021 rok istnieją. Jeśli nie, to trzeba będzie inaczej określić h
if test_data.empty:
    print("Brak danych na rok 2021 w pliku. Prognoza zostanie wygenerowana dla 365 dni.")
    n_periods = 365 # Liczba dni do prognozowania, jeśli brak danych na 2021
else:
    n_periods = len(test_data) # Liczba dni w zbiorze testowym 2021

print(f"Liczba punktów danych w zbiorze treningowym (2015-2020): {len(train_data)}")
print(f"Liczba punktów danych w zbiorze testowym (2021): {n_periods}")


print("\n--- 3.1 Model ARIMA ---")

# Task 8.1: Dobór parametrów (p, d, q) przy pomocy AIC/BIC
# Użycie auto_arima do automatycznego wyszukiwania najlepszych parametrów (p, d, q)
# trace=True wyświetla postęp i wyniki AIC/BIC dla każdej próby
# seasonal=False, bo zakładamy na razie brak sezonowości (można rozważyć w przyszłości SARIMA)
# suppress_warnings=True, żeby nie zaśmiecać konsoli ostrzeżeniami
# stepwise=True używa algorytmu stepwise, który jest szybszy
model_arima = pm.auto_arima(train_data,
                           start_p=1, start_q=1,
                           test='adf',    # Użycie testu ADF do określenia d
                           max_p=5, max_q=5,
                           m=1,           # m=1 dla danych niesezonowych (lub dla testu ADF)
                           d=None,        # Pozwala auto_arima określić d
                           seasonal=False, # Nie uwzględniamy sezonowości na tym etapie
                           trace=True,
                           error_action='ignore',
                           suppress_warnings=True,
                           stepwise=True)

print(f"\nNajlepsze parametry ARIMA (p, d, q) według auto_arima: {model_arima.order}")
print(f"Wartość AIC dla najlepszego modelu: {model_arima.aic()}")

# Task 8.2: Wytrenowanie modelu na danych 2015–2020 (już zrobione przez auto_arima, ale możemy stworzyć model od nowa dla jasności)
# Można też użyć modelu zwróconego przez auto_arima bezpośrednio
# history = [x for x in train_data] # Jeśli chcesz budować model iteracyjnie (niepotrzebne z pmdarima)

# Task 8.3: Generowanie prognozy na rok 2021
# Prognoza na 'n_periods' do przodu
forecast, conf_int = model_arima.predict(n_periods=n_periods, return_conf_int=True)

# Tworzenie indeksu dla prognozy (rok 2021)
# Zaczynamy od pierwszego dnia po danych treningowych
last_train_date = train_data.index[-1]
forecast_index = pd.date_range(start=last_train_date + pd.Timedelta(days=1), periods=n_periods, freq='D')
forecast_series = pd.Series(forecast, index=forecast_index)
conf_int_df = pd.DataFrame(conf_int, index=forecast_index, columns=['lower', 'upper'])


print("\nPrognoza na rok 2021 (pierwsze 5 dni):")
print(forecast_series.head())

# Wizualizacja wyników
plt.figure(figsize=(14, 7))
plt.plot(train_data.index, train_data, label='Dane treningowe (2015-2020)', color='blue')
if not test_data.empty:
    plt.plot(test_data.index, test_data, label='Dane rzeczywiste (2021)', color='green')
plt.plot(forecast_series.index, forecast_series, label='Prognoza ARIMA (2021)', color='red', linestyle='--')
plt.fill_between(conf_int_df.index, conf_int_df['lower'], conf_int_df['upper'], color='pink', alpha=0.3, label='Przedział ufności')
plt.title('Prognoza rynku walutowego EUR/USD modelem ARIMA')
plt.xlabel('Data')
plt.ylabel('Cena')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Ocena prognozy (jeśli mamy dane testowe na 2021)
if not test_data.empty:
    rmse = np.sqrt(mean_squared_error(test_data, forecast_series))
    print(f"\nRMSE dla prognozy ARIMA: {rmse:.4f}")
else:
    print("\nBrak danych rzeczywistych na rok 2021 do oceny RMSE.")