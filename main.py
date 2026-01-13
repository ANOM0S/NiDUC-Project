import matplotlib

matplotlib.use('Qt5Agg')

import matplotlib.pyplot as plt
import sensor_model
import voter_algorithms as va
import analysis_utils as au

if __name__ == '__main__':
    # --- WSPÓLNE PARAMETRY SYMULACJI ---
    TIME_POINTS = 200
    AMPLITUDE = 1.0
    BASE_NOISE = 0.05
    TOLERANCE_PLURALITY = 0.2

    # Lista scenariuszy do testowania:
    print(
        '''
        scenariusz 1: Idealne działanie z szumem bazowym
        scenariusz 2: Awaria pojedynczego sensora (Drift)
        scenariusz 3: Awaria większości sensorów (2 na 3)
        scenariusz 4: Awaria tzw. Szpilki (Impulsy)
        scenariusz 5: Duży szum na wszystkich
        scenariusz 6: Nierówne wagi (1 Główny + 2 Słabsze) - Awaria Głównego
        '''
    )
    scenario = int(input("Wybierz scenariusz (1 - 6): "))

    match scenario:
        case 1:
            # --- SCENARIUSZ 1: IDEALNE DZIAŁANIE (BASELINE) ---
            print("--- SCENARIUSZ 1: Działanie Idealne (Tylko Szum Bazowy) ---")
            NUM_SENSORS = 3
            # Wagi początkowe (dla 3 sensorów po równo to ok. 0.33)
            STATIC_WEIGHTS = [1 / 3, 1 / 3, 1 / 3]
            DRIFTING_SENSORS = []
            test_scenarios = []  # Brak wstrzykniętych awarii

        case 2:
            # --- SCENARIUSZ 2: POJEDYNCZA AWARIA (DRIFT) ---
            # Testuje, czy votery potrafią zignorować jeden "odjeżdżający" sensor.
            print("--- SCENARIUSZ 2: Awaria pojedynczego sensora (Drift) ---")
            NUM_SENSORS = 3
            STATIC_WEIGHTS = [1 / 3, 1 / 3, 1 / 3]
            DRIFTING_SENSORS = [2]  # Oznaczamy sensor nr 2 jako wadliwy (dla wykresów)

            test_scenarios = [
                {'sensor_index': 2, 'fault_type': 'drift', 'params': {'drift_rate': 0.02}}
            ]

        case 3:
            # --- SCENARIUSZ 3: AWARIA WIĘKSZOŚCI (2 z 3) ---
            # To jest test krytyczny - system powinien zawieść, chyba że voter ma pamięć (Wygładzający).
            print("--- SCENARIUSZ 3: Awaria większości (2 na 3 sensory wadliwe) ---")
            NUM_SENSORS = 3
            STATIC_WEIGHTS = [1 / 3, 1 / 3, 1 / 3]
            DRIFTING_SENSORS = [1, 2]

            test_scenarios = [
                {'sensor_index': 1, 'fault_type': 'gaussian', 'params': {'mean': 0.5, 'std': 0.5}},
                {'sensor_index': 2, 'fault_type': 'drift', 'params': {'drift_rate': -0.03}}
            ]

        case 4:
            # --- SCENARIUSZ 4: SZPILKI (OUTLIERS) ---
            # Idealny test dla algorytmu "N z M" (odrzucanie skrajnych) oraz Plurality.
            print("--- SCENARIUSZ 4: Zakłócenia impulsowe (Szpilki) ---")
            NUM_SENSORS = 3
            STATIC_WEIGHTS = [1 / 3, 1 / 3, 1 / 3]
            DRIFTING_SENSORS = [0]

            test_scenarios = [
                {'sensor_index': 0, 'fault_type': 'outlier', 'params': {'min_val': -2.0, 'max_val': 2.0, 'prob': 0.15}}
            ]

        case 5:
            # --- SCENARIUSZ 5: DUŻY SZUM WSZĘDZIE ---
            # Testuje jak algorytm Wygładzający radzi sobie z ogólnym chaosem.
            print("--- SCENARIUSZ 5: Wysoki poziom szumu na wszystkich sensorach ---")
            NUM_SENSORS = 3
            STATIC_WEIGHTS = [1 / 3, 1 / 3, 1 / 3]
            DRIFTING_SENSORS = [0, 1, 2]

            test_scenarios = [
                {'sensor_index': 0, 'fault_type': 'gaussian', 'params': {'mean': 0.0, 'std': 0.3}},
                {'sensor_index': 1, 'fault_type': 'gaussian', 'params': {'mean': 0.0, 'std': 0.4}},
                {'sensor_index': 2, 'fault_type': 'gaussian', 'params': {'mean': 0.0, 'std': 0.2}}
            ]

        case 6:
            print("--- SCENARIUSZ 6: Nierówne wagi [0.6, 0.2, 0.2] - Awaria Sensora Głównego ---")
            NUM_SENSORS = 3

            # Główny ma 60% głosu, pomocnicze po 20%
            STATIC_WEIGHTS = [0.6, 0.2, 0.2]

            DRIFTING_SENSORS = [0]  # Awaria dotyczy sensora z największą wagą!

            test_scenarios = [
                # Sensor 0 (Master) zaczyna powoli "odpływać"
                {'sensor_index': 0, 'fault_type': 'drift', 'params': {'drift_rate': 0.04}},

                # Sensor 1 (Slave) - działa poprawnie, ale ma większy szum (bo jest tańszy)
                {'sensor_index': 1, 'fault_type': 'gaussian', 'params': {'mean': 0.0, 'std': 0.15}},

                # Sensor 2 (Slave) - działa poprawnie, ale ma większy szum
                {'sensor_index': 2, 'fault_type': 'gaussian', 'params': {'mean': 0.0, 'std': 0.15}}
            ]

        case _:
            print("Niepoprawny wybór, uruchamiam scenariusz domyślny (1)")
            NUM_SENSORS = 3
            STATIC_WEIGHTS = [1 / 3, 1 / 3, 1 / 3]
            DRIFTING_SENSORS = []
            test_scenarios = []

    # --- URUCHOMIENIE SYMULACJI (WSPÓLNE DLA WSZYSTKICH SCENARIUSZY) ---
    # Generowanie danych
    nominal, data, time = sensor_model.generate_sensor_data(
        TIME_POINTS, NUM_SENSORS,
        amplitude=AMPLITUDE,
        base_noise_level=BASE_NOISE,
        fault_scenarios=test_scenarios
    )

    # Obliczenie wyników algorytmów
    voter_plurality_result = va.formalized_plurality_voter(data, tolerance=TOLERANCE_PLURALITY)
    voter_weighted_result = va.weighted_average_voter(data, weights=STATIC_WEIGHTS)

    # ----------------------------------------------------
    # WYNIKI LICZBOWE I ANALIZA BŁĘDÓW (ANALYSIS_UTILS)
    # ----------------------------------------------------
    au.display_numerical_results(nominal, data, voter_plurality_result, voter_weighted_result, time, num_samples=5)
    au.calculate_and_display_mse(nominal, voter_plurality_result, voter_weighted_result)

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
    plt.plot(time, data[i, :], linestyle, alpha=0.6, label=f'Sensor {i} (Waga: {STATIC_WEIGHTS[i]:.2f})')

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

plt.figure(figsize=(15, 8))
plt.subplot(2, 1, 1)

plt.plot(time, error_plurality, 'r--', label='Błąd Plurality Voter (e_P)', alpha=0.7)
plt.plot(time, error_weighted, 'b:', label='Błąd Weighted Average Voter (e_W)', alpha=0.7)
plt.axhline(0, color='k', linestyle='-', linewidth=1)  # Linia zero dla referencji
plt.title('Błąd Estymacji w Funkcji Czasu (Error Signal)', fontsize=14)
plt.xlabel('Czas')
plt.ylabel('Błąd (Nominalny - Voter)')
plt.grid(True)
plt.legend()

plt.subplot(2, 1, 2)

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
