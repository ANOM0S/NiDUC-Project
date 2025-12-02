import matplotlib
matplotlib.use('Qt5Agg')

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error


def generate_sensor_data(time_points, num_sensors, amplitude=1.0, frequency=1.0, noise_type='normal', noise_level=0.1,
                         drift_sensors=None):
    """
    Generuje nominalny sygnał sinusoidalny i zaszumione dane z sensorów.

    :param time_points: Liczba punktów czasowych (długość sygnału).
    :param num_sensors: Liczba symulowanych sensorów.
    :param amplitude: Amplituda sinusa.
    :param frequency: Częstotliwość sinusa.
    :param noise_type: Rodzaj zakłócenia ('normal', 'bias', 'spikes').
    :param noise_level: Poziom zakłóceń.
    :param drift_sensors: Lista indeksów sensorów, które mają stałe zakłócenie (dryft).
    :return: (sygnał nominalny, macierz danych z sensorów)
    """
    # 1. Nominalny sygnał sinusoidalny
    t = np.linspace(0, 2 * np.pi, time_points)  # Czas od 0 do 2*pi
    nominal_signal = amplitude * np.sin(frequency * t)

    sensor_data = np.zeros((num_sensors, time_points))

    # 2. Generowanie danych z sensorów z zakłóceniami
    for i in range(num_sensors):

        # Zakłócenie podstawowe (szum losowy)
        random_noise = 0
        if noise_type == 'normal':
            random_noise = np.random.normal(0, noise_level, time_points)
        elif noise_type == 'spikes':
            # Symulacja krótkich, silnych zakłóceń
            spikes = np.zeros(time_points)
            spike_indices = np.random.choice(time_points, size=int(time_points * noise_level * 0.1), replace=False)
            spikes[spike_indices] = np.random.uniform(2 * amplitude, 4 * amplitude, len(spike_indices))
            random_noise = np.random.normal(0, noise_level * 0.1, time_points) + spikes

        # Dodawanie dryfu (stałej wartości) do wybranych sensorów (symulacja awarii/kalibracji)
        bias_noise = 0
        if drift_sensors is not None and i in drift_sensors:
            bias_noise = np.full(time_points, noise_level * 3)  # stałe przesunięcie

        sensor_data[i, :] = nominal_signal + random_noise + bias_noise

    return nominal_signal, sensor_data, t


def formalized_plurality_voter(sensor_outputs, tolerance=0.1):
    """
    Implementacja Formalized Plurality Voter.
    Wybiera wartość, która ma najwięcej sąsiadów w ramach progu tolerancji.
    """
    consensus_value = np.zeros_like(sensor_outputs[0])

    for k in range(len(sensor_outputs[0])):
        # Wyniki ze wszystkich sensorów w danym punkcie czasowym k
        current_readings = sensor_outputs[:, k]
        max_votes = -1
        chosen_value = current_readings[0]  # Domyślna wartość

        # Sprawdzamy każdy odczyt jako potencjalnego kandydata
        for candidate in current_readings:
            # Liczymy "głosy" - ile innych sensorów jest w promieniu tolerancji
            votes = np.sum(np.abs(current_readings - candidate) <= tolerance)

            if votes > max_votes:
                max_votes = votes
                chosen_value = candidate
            elif votes == max_votes:
                # W przypadku remisu można np. wybrać wartość środkową
                pass

        consensus_value[k] = chosen_value

    return consensus_value


def weighted_average_voter(sensor_outputs, weights=None):
    """
    Implementacja Weighted Average Voter.
    Oblicza średnią ważoną wszystkich odczytów.
    """
    num_sensors = sensor_outputs.shape[0]

    # Domyślne wagi - równe dla wszystkich
    if weights is None:
        weights = np.ones(num_sensors) / num_sensors

    if np.sum(weights) != 1.0:
        weights = weights / np.sum(weights)  # Normalizacja wag

    # Obliczanie średniej ważonej dla każdego punktu czasowego
    # sum(W_i * X_i) / sum(W_i) (gdzie W to waga, X to odczyt)
    weighted_sum = np.dot(weights, sensor_outputs)

    return weighted_sum


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


# ----------------------------------------------------
# PARAMETRY SYMULACJI
# ----------------------------------------------------
NUM_SENSORS = 7
TIME_POINTS = 200
DRIFTING_SENSORS = [0, 6]  # Sensor 0 i 6 mają dryft (stałe zakłócenie)
WEIGHTS = [0.1, 0.15, 0.2, 0.15, 0.2, 0.15, 0.05]  # Wagi dla 7 sensorów, suma musi dać 1.0

# Generowanie danych
nominal, data, time = generate_sensor_data(
    TIME_POINTS, NUM_SENSORS,
    noise_type='normal',
    noise_level=0.1,
    drift_sensors=DRIFTING_SENSORS
)

# Obliczenie wyników algorytmów
voter_plurality_result = formalized_plurality_voter(data, tolerance=0.2)
voter_weighted_result = weighted_average_voter(data, weights=WEIGHTS)

# ----------------------------------------------------
# WYNIKI LICZBOWE
# ----------------------------------------------------
display_numerical_results(nominal, data, voter_plurality_result, voter_weighted_result, time, num_samples=5)

# ----------------------------------------------------
# ANALIZA BŁĘDÓW
# ----------------------------------------------------
calculate_and_display_mse(nominal, voter_plurality_result, voter_weighted_result)

# ----------------------------------------------------
# WIZUALIZACJA (Wykresy)
# ----------------------------------------------------
plt.figure(figsize=(15, 8))

# 1. Wykres wszystkich odczytów i sygnału nominalnego
plt.subplot(2, 1, 1)  # Tworzy siatkę 2 wiersze, 1 kolumna, wykres 1
plt.plot(time, nominal, 'k-', linewidth=3, label='Sygnał Nominalny (Idealny)')
for i in range(NUM_SENSORS):
    # Podkreślenie dryfujących sensorów
    linestyle = '--' if i in DRIFTING_SENSORS else '-'
    plt.plot(time, data[i, :], linestyle, alpha=0.6, label=f'Sensor {i} (Waga: {WEIGHTS[i]:.2f})')

plt.title(f'Odczyty z {NUM_SENSORS} Sensorów (Zakłócenie Normalne + Dryft Sensorów {DRIFTING_SENSORS})', fontsize=14)
plt.xlabel('Czas')
plt.ylabel('Amplituda')
plt.grid(True)
plt.legend(loc='upper right', ncol=3)

# 2. Wykres Porównanie Wyników Głosowania
plt.subplot(2, 1, 2)  # Wykres 2
plt.plot(time, nominal, 'k-', linewidth=4, label='Sygnał Nominalny (Idealny)')
plt.plot(time, voter_plurality_result, 'r--', linewidth=2, label='Formalized Plurality Voter (Tol=0.2)')
plt.plot(time, voter_weighted_result, 'b:', linewidth=2, label='Weighted Average Voter')

plt.title('Porównanie Wyników Algorytmów Głosowania', fontsize=14)
plt.xlabel('Czas')
plt.ylabel('Amplituda')
plt.grid(True)
plt.legend(loc='upper right')
plt.tight_layout()  # Automatycznie dopasowuje odstępy
plt.show()

# Obliczenie sygnałów błędu
error_plurality = nominal - voter_plurality_result
error_weighted = nominal - voter_weighted_result

plt.figure(figsize=(15, 5))
plt.plot(time, error_plurality, 'r--', label='Błąd Plurality Voter (e_P)', alpha=0.7)
plt.plot(time, error_weighted, 'b:', label='Błąd Weighted Average Voter (e_W)', alpha=0.7)
plt.axhline(0, color='k', linestyle='-', linewidth=1)  # Linia zero dla referencji
plt.title('Błąd Estymacji w Funkcji Czasu (Error Signal)', fontsize=14)
plt.xlabel('Czas')
plt.ylabel('Błąd (Nominalny - Voter)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(15, 5))

# Histogram dla Plurality Voter
plt.hist(error_plurality, bins=50, alpha=0.6, label='Plurality Voter', color='red', density=True)

# Histogram dla Weighted Average Voter
plt.hist(error_weighted, bins=50, alpha=0.6, label='Weighted Average Voter', color='blue', density=True)

plt.title('Histogram Rozkładu Błędu Estymacji', fontsize=14)
plt.xlabel('Wielkość Błędu (e)')
plt.ylabel('Gęstość')
plt.legend()
plt.grid(axis='y', alpha=0.5)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 8))

# Rysujemy linię idealną y=x
min_val = np.min(nominal)
max_val = np.max(nominal)
plt.plot([min_val, max_val], [min_val, max_val], 'k--', label='Linia Idealna (y=x)', alpha=0.8)

# Wykres punktowy dla Plurality Voter
plt.scatter(nominal, voter_plurality_result, c='red', s=10, alpha=0.5, label='Plurality Voter')

# Wykres punktowy dla Weighted Average Voter
plt.scatter(nominal, voter_weighted_result, c='blue', s=10, alpha=0.5, label='Weighted Average Voter')

plt.title('Wartość Nominalna vs Wartość Oszacowana', fontsize=14)
plt.xlabel('Wartość Nominalna')
plt.ylabel('Wartość Oszacowana (Voter)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
