import numpy as np
from sklearn.metrics import mean_squared_error
import pandas as pd

def display_numerical_results(nominal, sensor_data, voter_plurality, voter_weighted, time, num_samples=5):
    """
    Wyświetla wyniki liczbowe dla kilku losowo wybranych punktów czasowych.
    """
    time_points = len(nominal)
    # Wybieramy losowe punkty czasowe do analizy
    sample_indices = np.random.choice(time_points, size=num_samples, replace=False)
    sample_indices.sort()

    data_dict = {
        'T_index': sample_indices,
        'Czas (t)': np.round(time[sample_indices], 2),
        'Nominalny': np.round(nominal[sample_indices], 3),
    }

    # Dodajemy wyniki z każdego sensora
    for i in range(sensor_data.shape[0]):
        data_dict[f'Sensor {i}'] = np.round(sensor_data[i, sample_indices], 3)

    # Dodajemy wyniki z voterów
    data_dict['Voter Plurality'] = np.round(voter_plurality[sample_indices], 3)
    data_dict['Voter Weighted'] = np.round(voter_weighted[sample_indices], 3)

    df = pd.DataFrame(data_dict)
    print("\n" + "=" * 80)
    print(f"TABELA WYNIKÓW DLA {num_samples} LOSOWYCH PUNKTÓW CZASOWYCH")
    print("=" * 80)
    print(df.to_markdown(index=False))  # Używamy markdown, żeby dobrze wyglądało w konsoli
    print("=" * 80)


def calculate_and_display_mse(nominal, voter_plurality, voter_weighted):
    """
    Oblicza i wyświetla błąd średniokwadratowy (MSE) dla obu algorytmów.
    """
    # Obliczenie MSE
    mse_plurality = mean_squared_error(nominal, voter_plurality)
    mse_weighted = mean_squared_error(nominal, voter_weighted)

    print("\n" + "=" * 40)
    print("ANALIZA BŁĘDÓW (MSE)")
    print("=" * 40)
    print(f"MSE Formalized Plurality Voter:  {mse_plurality:.6f}")
    print(f"MSE Weighted Average Voter:      {mse_weighted:.6f}")

    # Małe podsumowanie
    if mse_plurality < mse_weighted:
        print("\nFormalized Plurality Voter lepiej radził sobie z tymi zakłóceniami.")
    else:
        print("\nWeighted Average Voter lepiej radził sobie z tymi zakłóceniami.")
    print("=" * 40)