import numpy as np
from sklearn.metrics import mean_squared_error
import pandas as pd


def display_numerical_results(nominal, sensor_data, res_plurality, res_weighted, res_nzm, res_smoothing, res_dynamic,
                              time, num_samples=5):
    """
    Wyświetla tabelę wyników dla wszystkich 5 algorytmów.
    """
    time_points = len(nominal)
    # Zabezpieczenie, gdyby próbek było mniej niż num_samples
    actual_samples = min(num_samples, time_points)
    sample_indices = np.random.choice(time_points, size=actual_samples, replace=False)
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
    data_dict['Dynamic'] = np.round(res_dynamic[sample_indices], 3)

    df = pd.DataFrame(data_dict)
    print("\n" + "=" * 100)
    print(f"TABELA PORÓWNAWCZA (Losowe próbki)")
    print("=" * 100)
    print(df.to_string(index=False))
    print("=" * 100)


def calculate_and_display_mse(nominal, res_plurality, res_weighted, res_nzm, res_smoothing, res_dynamic):
    """
    Oblicza i wyświetla błąd średniokwadratowy (MSE) dla 5 algorytmów.
    """
    mse_p = mean_squared_error(nominal, res_plurality)
    mse_w = mean_squared_error(nominal, res_weighted)
    mse_n = mean_squared_error(nominal, res_nzm)
    mse_s = mean_squared_error(nominal, res_smoothing)
    mse_d = mean_squared_error(nominal, res_dynamic)

    print("\n" + "=" * 50)
    print("ANALIZA BŁĘDÓW (MSE - Im mniej tym lepiej)")
    print("=" * 50)
    print(f"Plurality Voter:         {mse_p:.6f}")
    print(f"Weighted Voter (Stat):   {mse_w:.6f}")
    print(f"N-z-M Voter:             {mse_n:.6f}")
    print(f"Smoothing Voter:         {mse_s:.6f}")
    print(f"Dynamic Weighted (Brøn): {mse_d:.6f}")
    print("-" * 50)

    # Wyłonienie zwycięzcy
    results = {
        'Plurality': mse_p,
        'Weighted (Static)': mse_w,
        'N-z-M': mse_n,
        'Smoothing': mse_s,
        'Dynamic (Brøn)': mse_d
    }
    winner = min(results, key=results.get)
    print(f"🏆 NAJLEPSZY ALGORYTM: {winner}")
    print("=" * 50)