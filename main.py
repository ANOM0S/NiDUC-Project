import matplotlib
matplotlib.use('Qt5Agg')

import matplotlib.pyplot as plt
import sensor_model
import voter_algorithms as va
import analysis_utils as au


# ----------------------------------------------------
# PARAMETRY SYMULACJI
# ----------------------------------------------------
NUM_SENSORS = 7
TIME_POINTS = 200
DRIFTING_SENSORS = [0, 6]  # Sensor 0 i 6 mają dryft (stałe zakłócenie)
WEIGHTS = [0.1, 0.15, 0.2, 0.15, 0.2, 0.15, 0.05]  # Wagi dla 7 sensorów, suma musi dać 1.0

# Generowanie danych
nominal, data, time = sensor_model.generate_sensor_data(
    TIME_POINTS, NUM_SENSORS,
    noise_type='normal',
    noise_level=0.1,
    drift_sensors=DRIFTING_SENSORS
)

# Obliczenie wyników algorytmów
voter_plurality_result = va.formalized_plurality_voter(data, tolerance=0.2)
voter_weighted_result = va.weighted_average_voter(data, weights=WEIGHTS)

# ----------------------------------------------------
# WYNIKI LICZBOWE
# ----------------------------------------------------
au.display_numerical_results(nominal, data, voter_plurality_result, voter_weighted_result, time, num_samples=5)

# ----------------------------------------------------
# ANALIZA BŁĘDÓW
# ----------------------------------------------------
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
