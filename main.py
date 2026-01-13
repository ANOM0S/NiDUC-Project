import matplotlib
# Jeśli masz błąd z Qt5Agg, zakomentuj poniższą linię:
matplotlib.use('Qt5Agg')

import matplotlib.pyplot as plt
import numpy as np
import sensor_model
import voter_algorithms as va
import analysis_utils as au

if __name__ == '__main__':
    # --- WSPÓLNE PARAMETRY ---
    TIME_POINTS = 300
    AMPLITUDE = 1.0
    BASE_NOISE = 0.05
    TOLERANCE_PLURALITY = 0.3

    # Menu wyboru
    print(
        '''
        scenariusz 1: Idealne działanie
        scenariusz 2: Awaria pojedynczego sensora (Drift)
        scenariusz 3: Awaria większości sensorów (2 na 3)
        scenariusz 4: Szpilki (Test dla N-z-M)
        scenariusz 5: Duży szum wszędzie (Test dla Smoothing)
        scenariusz 6: Nierówne wagi (Awaria Głównego)
        '''
    )
    try:
        scenario = int(input("Wybierz scenariusz (1 - 6): "))
    except ValueError:
        scenario = 1

    # --- KONFIGURACJA SCENARIUSZY ---
    match scenario:
        case 1:
            print("--- SCENARIUSZ 1: Baseline ---")
            NUM_SENSORS = 3
            STATIC_WEIGHTS = [1 / 3, 1 / 3, 1 / 3]
            DRIFTING_SENSORS = []
            test_scenarios = []

        case 2:
            print("--- SCENARIUSZ 2: Drift jednego sensora ---")
            NUM_SENSORS = 3
            STATIC_WEIGHTS = [1 / 3, 1 / 3, 1 / 3]
            DRIFTING_SENSORS = [2]
            test_scenarios = [{'sensor_index': 2, 'fault_type': 'drift', 'params': {'drift_rate': 0.03}}]

        case 3:
            print("--- SCENARIUSZ 3: Awaria 2 z 3 sensorów ---")
            NUM_SENSORS = 3
            STATIC_WEIGHTS = [1 / 3, 1 / 3, 1 / 3]
            DRIFTING_SENSORS = [1, 2]
            test_scenarios = [
                {'sensor_index': 1, 'fault_type': 'gaussian', 'params': {'mean': 0.5, 'std': 0.5}},
                {'sensor_index': 2, 'fault_type': 'drift', 'params': {'drift_rate': -0.04}}
            ]

        case 4:
            print("--- SCENARIUSZ 4: Szpilki (Outliers) ---")
            NUM_SENSORS = 3
            STATIC_WEIGHTS = [1 / 3, 1 / 3, 1 / 3]
            DRIFTING_SENSORS = [0]
            test_scenarios = [
                {'sensor_index': 0, 'fault_type': 'outlier', 'params': {'min_val': -3.0, 'max_val': 3.0, 'prob': 0.1}}
            ]

        case 5:
            print("--- SCENARIUSZ 5: Wysoki szum (Chaos) ---")
            NUM_SENSORS = 3
            STATIC_WEIGHTS = [1 / 3, 1 / 3, 1 / 3]
            DRIFTING_SENSORS = [0, 1, 2]
            test_scenarios = [
                {'sensor_index': 0, 'fault_type': 'gaussian', 'params': {'mean': 0.0, 'std': 0.3}},
                {'sensor_index': 1, 'fault_type': 'gaussian', 'params': {'mean': 0.0, 'std': 0.4}},
                {'sensor_index': 2, 'fault_type': 'gaussian', 'params': {'mean': 0.0, 'std': 0.3}}
            ]

        case 6:
            print("--- SCENARIUSZ 6: Heterogeniczny (Awaria Mastera) ---")
            NUM_SENSORS = 3
            STATIC_WEIGHTS = [0.6, 0.2, 0.2]
            DRIFTING_SENSORS = [0]
            test_scenarios = [
                {'sensor_index': 0, 'fault_type': 'drift', 'params': {'drift_rate': 0.05}},
                {'sensor_index': 1, 'fault_type': 'gaussian', 'params': {'mean': 0.0, 'std': 0.1}},
                {'sensor_index': 2, 'fault_type': 'gaussian', 'params': {'mean': 0.0, 'std': 0.1}}
            ]

        case _:
            NUM_SENSORS = 3
            STATIC_WEIGHTS = [1 / 3, 1 / 3, 1 / 3]
            DRIFTING_SENSORS = []
            test_scenarios = []

    # --- GENEROWANIE DANYCH ---
    nominal, data, time = sensor_model.generate_sensor_data(
        TIME_POINTS, NUM_SENSORS,
        amplitude=AMPLITUDE,
        base_noise_level=BASE_NOISE,
        fault_scenarios=test_scenarios
    )

    # --- OBLICZENIA VOTERÓW ---

    # 1. Plurality (Wektorowo)
    res_plurality = va.formalized_plurality_voter(data, tolerance=TOLERANCE_PLURALITY)

    # 2. Weighted Static (Wektorowo - wagi stałe)
    res_weighted_static = va.weighted_average_voter(data, weights=STATIC_WEIGHTS)

    # Inicjalizacja list dla algorytmów liczonych w pętli
    res_nzm = []
    res_smoothing = []
    res_dynamic_weighted = []  # <--- NOWA LISTA

    prev_val_smooth = None

    for t_idx in range(TIME_POINTS):
        readings = data[:, t_idx]

        # 3. N z M
        val_nzm = va.n_z_m_voter(readings, m_to_keep=2)
        res_nzm.append(val_nzm)

        # 4. Smoothing
        val_smooth = va.smoothing_voter(readings, prev_val_smooth, alpha=0.15)
        prev_val_smooth = val_smooth
        res_smoothing.append(val_smooth)

        # 5. DYNAMIC WEIGHTED (Brøn) <--- TU DODAJEMY WYWOŁANIE
        # Parametr 'a' decyduje o czułości. Mniejsze 'a' = mocniejsze karanie za błędy.
        val_dynamic = va.weighted_average_dynamic_bron(readings, a=0.5)
        res_dynamic_weighted.append(val_dynamic)

    # Konwersja na numpy array
    res_nzm = np.array(res_nzm)
    res_smoothing = np.array(res_smoothing)
    res_dynamic_weighted = np.array(res_dynamic_weighted)

    # --- ANALIZA WYNIKÓW (Możesz zaktualizować funkcje w analysis_utils, żeby przyjmowały 5 argumentów, albo pominąć to tutaj) ---
    # au.calculate_and_display_mse(...)

    # --- WYKRESY ---
    plt.figure(figsize=(14, 10))

    # Panel 1: Sensory
    plt.subplot(2, 1, 1)
    plt.plot(time, nominal, 'k--', linewidth=2, label='Nominalny', alpha=0.8)
    for i in range(NUM_SENSORS):
        style = ':' if i in DRIFTING_SENSORS else '-'
        plt.plot(time, data[i, :], linestyle=style, alpha=0.5, label=f'Sensor {i}')
    plt.title(f'Dane z sensorów (Scenariusz {scenario})')
    plt.legend()
    plt.grid(True)

    # Panel 2: Wyniki algorytmów
    plt.subplot(2, 1, 2)
    plt.plot(time, nominal, 'k--', linewidth=3, alpha=0.3, label='Nominalny')

    # Rysujemy nasze 5 algorytmów
    plt.plot(time, res_plurality, label='Plurality', linewidth=1)
    plt.plot(time, res_weighted_static, label='Weighted (Static)', linewidth=1)
    plt.plot(time, res_nzm, label='N-z-M', linewidth=1)
    plt.plot(time, res_smoothing, label='Smoothing', linewidth=1)

    # Nowy algorytm na wykresie:
    plt.plot(time, res_dynamic_weighted, 'r--', linewidth=2, label='Dynamic Weighted (Brøn)')

    plt.title('Porównanie 5 Algorytmów')
    plt.xlabel('Czas')
    plt.legend(loc='upper right')
    plt.grid(True)

    plt.tight_layout()
    plt.show()