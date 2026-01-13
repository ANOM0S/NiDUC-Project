import numpy as np
from sklearn.metrics import mean_squared_error
import pandas as pd


def display_numerical_results(nominal, sensor_data, res_plurality, res_weighted, res_nzm, res_smoothing, time,
                              num_samples=5):
    """
    Wyświetla tabelę wyników dla wszystkich 4 algorytmów.
    """
    time_points = len(nominal)
    sample_indices = np.random.choice(time_points, size=num_samples, replace=False)
    sample_indices.sort()

    data_dict = {
        'T': np.round(time[sample_indices], 2),
        'Nominal': np.round(nominal[sample_indices], 3),
    }

    # Wyniki algorytmów
    data_dict['Plurality'] = np.round(res_plurality[sample_indices], 3)
    data_dict['Weighted'] = np.round(res_weighted[sample_indices], 3)
    data_dict['N-z-M'] = np.round(res_nzm[sample_indices], 3)
    data_dict['Smooth'] = np.round(res_smoothing[sample_indices], 3)

    df = pd.DataFrame(data_dict)
    print("\n" + "=" * 80)
    print(f"TABELA PORÓWNAWCZA (Losowe próbki)")
    print("=" * 80)
    print(df.to_string(index=False))
    print("=" * 80)


def calculate_and_display_mse(nominal, res_plurality, res_weighted, res_nzm, res_smoothing):
    """
    Oblicza i wyświetla błąd średniokwadratowy (MSE) dla 4 algorytmów.
    """
    mse_p = mean_squared_error(nominal, res_plurality)
    mse_w = mean_squared_error(nominal, res_weighted)
    mse_n = mean_squared_error(nominal, res_nzm)
    mse_s = mean_squared_error(nominal, res_smoothing)

    print("\n" + "=" * 40)
    print("ANALIZA BŁĘDÓW (MSE - Im mniej tym lepiej)")
    print("=" * 40)
    print(f"Plurality Voter:   {mse_p:.6f}")
    print(f"Weighted Voter:    {mse_w:.6f}")
    print(f"N-z-M Voter:       {mse_n:.6f}")
    print(f"Smoothing Voter:   {mse_s:.6f}")
    print("-" * 40)

    # Wyłonienie zwycięzcy
    results = {'Plurality': mse_p, 'Weighted': mse_w, 'N-z-M': mse_n, 'Smoothing': mse_s}
    winner = min(results, key=results.get)
    print(f"🏆 NAJLEPSZY ALGORYTM: {winner}")
    print("=" * 40)