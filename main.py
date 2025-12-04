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
        scenariusz 2: 2 z 5 działają wadliwie
        scenariusz 3: Awaria większości sensorów 4 na 6
        scenariusz 4: Awaria tzw. Szpilki
        scenariusz 5: Awaria krytyczna 6 na 6
        '''
    )
    scenario = int(input("Wybierz scenariusz (1 - 5): "))

    match scenario:
        case 1:
            # ----------------------------------------------------
            # SCENARIUSZ A: Działanie Idealne (Baseline)
            # ----------------------------------------------------

            print("--- SCENARIUSZ A: Działanie Idealne (Tylko Szum Bazowy) ---")
            NUM_SENSORS = 5
            STATIC_WEIGHTS = [0.2, 0.2, 0.2, 0.2, 0.2] # Równe wagi
            DRIFTING_SENSORS = [] # Brak awarii do podświetlenia

            test_scenarios = [] # Brak wstrzykniętych awarii

        case 2:
            # ----------------------------------------------------
            # SCENARIUSZ B: Awarie Mniej niż 50% (Dryft + Stuck-At)
            # ----------------------------------------------------

            print("--- SCENARIUSZ B: Realistyczny (2 z 5 sensorów z awarią) ---")
            NUM_SENSORS = 5
            STATIC_WEIGHTS = [0.1, 0.3, 0.3, 0.2, 0.1]  # Wagi celowo zróżnicowane
            DRIFTING_SENSORS = [0, 4]

            test_scenarios = [
                # Sensor 0: stały duży dryft (Bias)
                {'type': 'bias', 'sensors': [0], 'magnitude': 2.0},
                # Sensor 4: Awaria Stuck-At-Value (Zwraca stałą 0.5)
                {'type': 'stuck', 'sensors': [4], 'value': 0.5}
            ]

        case 3:
            # ----------------------------------------------------
            # SCENARIUSZ C: Awaria Większości (4 z 6 sensorów z awarią)
            # ----------------------------------------------------

            print("--- SCENARIUSZ C: Awaria Większości (Tylko 2/6 Poprawne) ---")
            NUM_SENSORS = 6
            STATIC_WEIGHTS = [0.15, 0.15, 0.15, 0.20, 0.20, 0.15]
            DRIFTING_SENSORS = [0, 1, 2, 5]

            test_scenarios = [
                # Sensor 0 i 1: duży dryft w przeciwnych kierunkach
                {'type': 'bias', 'sensors': [0], 'magnitude': 2.5},
                {'type': 'bias', 'sensors': [1], 'magnitude': -2.5},
                # Sensor 2: Awaria Stuck-At-Value (Zwraca stałą 0.0)
                {'type': 'stuck', 'sensors': [2], 'value': 0.0},
                # Sensor 5: Awaria Freeze od połowy
                {'type': 'freeze', 'sensors': [5], 'time_point': 100}
            ]

        case 4:
            # ----------------------------------------------------
            # SCENARIUSZ D: Szpilki (Test wrażliwości na pojedyncze błędy)
            # ----------------------------------------------------

            print("--- SCENARIUSZ D: Zakłócenia Impulsowe (Spikes) ---")
            NUM_SENSORS = 5
            STATIC_WEIGHTS = [0.2, 0.2, 0.2, 0.2, 0.2]
            DRIFTING_SENSORS = [1, 3]

            test_scenarios = [
                # Sensor 1 i 3: sporadyczne, silne szpilki
                {'type': 'spikes', 'sensors': [1, 3], 'magnitude': 5.0, 'density': 0.01}
            ]
        case 5:
            # ----------------------------------------------------
            # SCENARIUSZ E: Awaria Krytyczna (Brak Konsensusu)
            # ----------------------------------------------------

            print("--- SCENARIUSZ E: Krytyczna (Dwie równe, duże grupy błędów) ---")
            NUM_SENSORS = 6
            STATIC_WEIGHTS = [0.1, 0.2, 0.2, 0.2, 0.2, 0.1]
            DRIFTING_SENSORS = [0, 1, 4, 5]

            test_scenarios = [
                # Grupa 1 (Sensory 0, 1): Duży dryft w górę
                {'type': 'bias', 'sensors': [0, 1], 'magnitude': 3.0},
                # Grupa 2 (Sensory 4, 5): Duży dryft w dół
                {'type': 'bias', 'sensors': [4, 5], 'magnitude': -3.0}
                # Sensory 2, 3: Pracują poprawnie, ale są w mniejszości!
            ]

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
