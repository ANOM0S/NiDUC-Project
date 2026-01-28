import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
import numpy as np
import os
import sensor_model
import voter_algorithms as va

# Parametry dla Smoothing Voter
SMOOTHING_EPSILON = 0.2  # Próg zgody dla większości
SMOOTHING_BETA = 0.35     # Próg ciągłości historycznej

# --- KONFIGURACJA wizualna ---
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.5
plt.rcParams['lines.linewidth'] = 1.5

# --- KONFIGURACJA FOLDERU NA WYNIKI ---
OUTPUT_DIR = 'wykresy_symulacja_pl'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    print(f"Utworzono folder na wyniki: {OUTPUT_DIR}")

if __name__ == '__main__':
    # --- WSPÓLNE PARAMETRY SYMULACJI ---
    TIME_POINTS = 300
    AMPLITUDE = 1.0
    BASE_NOISE = 0.05

    # --- PARAMETRY ALGORYTMÓW ---
    # Pluralny
    TOLERANCE_PLURALITY = 0.3

    # Zaawansowany M-z-N
    # Dla 3 sensorów po 0.33 wagi:
    # M powinno być np. 0.5 (żeby 2 sensory wystarczyły: 0.33+0.33 = 0.66 > 0.5)
    NZM_THRESHOLD_M = 0.5
    NZM_TOLERANCE_TAU = 0.4
    NZM_THRESHOLD_GAMMA = 0.5

    print(
        '''
        DOSTĘPNE SCENARIUSZE TESTOWE:
        -------------------------------------------------------
        1: Warunki Nominalne (Baseline) - Tylko szum
        2: Awaria Pojedyncza (Dryft) - Jeden czujnik odpływa
        3: Awaria Większości (2 z 3) - Dryft + Szum
        4: Wartości Odstające (Szpilki) - Impulsy losowe
        5: Wysoki Poziom Szumu (Chaos)
        6: System Heterogeniczny (Awaria Mastera)
        7: Uszkodzenia Twarde (Zacięcie + Bias)
        '''
    )
    try:
        scenario = int(input("Wybierz scenariusz (1 - 7): "))
    except ValueError:
        scenario = 1

    # --- KONFIGURACJA SCENARIUSZY ---
    match scenario:
        case 1:
            title_text = "Warunki Nominalne (Szum bazowy)"
            NUM_SENSORS = 3
            STATIC_WEIGHTS = [1 / 3, 1 / 3, 1 / 3]
            DRIFTING_SENSORS = []
            test_scenarios = []
        case 2:
            title_text = "Awaria Pojedyncza (Dryft)"
            NUM_SENSORS = 3
            STATIC_WEIGHTS = [1 / 3, 1 / 3, 1 / 3]
            DRIFTING_SENSORS = [2]
            test_scenarios = [{'sensor_index': 2, 'fault_type': 'drift', 'params': {'drift_rate': 0.03}}]
        case 3:
            title_text = "Awaria Większości (2 z 3)"
            NUM_SENSORS = 3
            STATIC_WEIGHTS = [1 / 3, 1 / 3, 1 / 3]
            DRIFTING_SENSORS = [1, 2]
            test_scenarios = [
                {'sensor_index': 1, 'fault_type': 'gaussian', 'params': {'mean': 0.5, 'std': 0.5}},
                {'sensor_index': 2, 'fault_type': 'drift', 'params': {'drift_rate': -0.04}}
            ]
        case 4:
            title_text = "Wartości Odstające (Szpilki)"
            NUM_SENSORS = 3
            STATIC_WEIGHTS = [1 / 3, 1 / 3, 1 / 3]
            DRIFTING_SENSORS = [0]
            test_scenarios = [
                {'sensor_index': 0, 'fault_type': 'outlier', 'params': {'min_val': -3.0, 'max_val': 3.0, 'prob': 0.1}}
            ]
        case 5:
            title_text = "Wysoki Poziom Szumu (Chaos)"
            NUM_SENSORS = 3
            STATIC_WEIGHTS = [1 / 3, 1 / 3, 1 / 3]
            DRIFTING_SENSORS = [0, 1, 2]
            test_scenarios = [
                {'sensor_index': 0, 'fault_type': 'gaussian', 'params': {'mean': 0.0, 'std': 0.3}},
                {'sensor_index': 1, 'fault_type': 'gaussian', 'params': {'mean': 0.0, 'std': 0.4}},
                {'sensor_index': 2, 'fault_type': 'gaussian', 'params': {'mean': 0.0, 'std': 0.3}}
            ]
        case 6:
            title_text = "System Heterogeniczny (Awaria Mastera)"
            NUM_SENSORS = 3
            STATIC_WEIGHTS = [0.6, 0.2, 0.2]  # Master (Czujnik 0) ma 60% wagi
            DRIFTING_SENSORS = [0]
            test_scenarios = [
                {'sensor_index': 0, 'fault_type': 'drift', 'params': {'drift_rate': 0.05}},
                {'sensor_index': 1, 'fault_type': 'gaussian', 'params': {'mean': 0.0, 'std': 0.1}},
                {'sensor_index': 2, 'fault_type': 'gaussian', 'params': {'mean': 0.0, 'std': 0.1}}
            ]
        case 7:
            title_text = "Uszkodzenia Twarde (Zacięcie + Bias)"
            NUM_SENSORS = 3
            STATIC_WEIGHTS = [1 / 3, 1 / 3, 1 / 3]
            DRIFTING_SENSORS = [1, 2]
            test_scenarios = [
                {'sensor_index': 1, 'fault_type': 'stuck', 'params': {'value': -0.8, 'start_idx': 50}},
                {'sensor_index': 2, 'fault_type': 'bias', 'params': {'offset': 0.5}}
            ]
        case _:
            title_text = "Domyślny"
            NUM_SENSORS = 3
            STATIC_WEIGHTS = [1 / 3, 1 / 3, 1 / 3]
            DRIFTING_SENSORS = []
            test_scenarios = []

    print(f"\n--- URUCHAMIAM SCENARIUSZ: {title_text} ---")

    # --- GENEROWANIE DANYCH ---
    nominal, data, time = sensor_model.generate_sensor_data(
        TIME_POINTS, NUM_SENSORS,
        amplitude=AMPLITUDE,
        base_noise_level=BASE_NOISE,
        fault_scenarios=test_scenarios
    )

    # --- OBLICZENIA VOTERÓW ---
    res_plurality = va.formalized_plurality_voter(data, tolerance=TOLERANCE_PLURALITY)
    res_weighted_static = va.weighted_average_voter(data, weights=STATIC_WEIGHTS)

    # --- OBLICZENIA VOTERÓW Z PAMIĘCIĄ ---
    res_nzm = []
    res_smoothing = []
    res_dynamic_weighted = []

    # Zmienne pamięci
    prev_val_smooth = 0.0
    prev_val_nzm = 0.0

    for t_idx in range(TIME_POINTS):
        readings = data[:, t_idx]

        # 1. Zaawansowany M-z-N
        val_nzm = va.advanced_m_out_of_n_voter(
            readings,
            weights=STATIC_WEIGHTS,
            threshold_m=NZM_THRESHOLD_M,
            threshold_tau=NZM_TOLERANCE_TAU,
            threshold_gamma=NZM_THRESHOLD_GAMMA,
            previous_result=prev_val_nzm
        )
        prev_val_nzm = val_nzm
        res_nzm.append(val_nzm)

        # 2. Wygładzający
        val_smooth = va.smoothing_voter(
            readings,
            previous_result=prev_val_smooth,
            threshold_majority_epsilon=SMOOTHING_EPSILON,
            threshold_smoothing_beta=SMOOTHING_BETA
        )
        prev_val_smooth = val_smooth
        res_smoothing.append(val_smooth)

        # 3. Dynamiczny (Brøn)
        res_dynamic_weighted.append(va.weighted_average_dynamic_bron(readings, a=0.5))

    # Konwersja na numpy array
    res_nzm = np.array(res_nzm)
    res_smoothing = np.array(res_smoothing)
    res_dynamic_weighted = np.array(res_dynamic_weighted)

    # --- WYLICZANIE BŁĘDÓW ---
    err_plurality = nominal - res_plurality
    err_weighted = nominal - res_weighted_static
    err_nzm = nominal - res_nzm
    err_smooth = nominal - res_smoothing
    err_dynamic = nominal - res_dynamic_weighted

    # =========================================================================
    # WYKRES 1: PRZEBIEGI CZASOWE (3 PANELE)
    # =========================================================================
    fig1, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
    fig1.subplots_adjust(top=0.93, hspace=0.3)
    fig1.suptitle(f"Scenariusz: {title_text}", fontsize=16, fontweight='bold')

    # PANEL A: DANE Z CZUJNIKÓW
    ax1.plot(time, nominal, 'k--', linewidth=2, label='Wzorzec (Idealny)')
    for i in range(NUM_SENSORS):
        label_text = f'Czujnik {i}'
        if i in DRIFTING_SENSORS:
            ax1.plot(time, data[i, :], color='red', alpha=0.6, linestyle=':', linewidth=2,
                     label=label_text + ' (Awaria)')
        else:
            ax1.plot(time, data[i, :], color='gray', alpha=0.5, linewidth=1, label=label_text)
    ax1.set_ylabel("Amplituda")
    ax1.set_title("A. Surowe dane z czujników", loc='left', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=9, framealpha=0.9)
    ax1.grid(True, linestyle='--', alpha=0.6)

    # PANEL B: ALGORYTMY PODSTAWOWE
    ax2.plot(time, nominal, 'k--', alpha=0.2)
    ax2.plot(time, res_plurality, color='blue', linewidth=1.5, label='Pluralny (Plurality)')
    ax2.plot(time, res_weighted_static, color='green', linewidth=1.5, label='Średnia Ważona (Statyczna)')
    ax2.plot(time, res_dynamic_weighted, color='crimson', linestyle='--', linewidth=1.5, label='Dynamiczny (Brøn)')
    ax2.set_ylabel("Wynik")
    ax2.set_title("B. Grupa: Średnie i Głosowanie", loc='left', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=9, framealpha=0.9)
    ax2.grid(True, linestyle='--', alpha=0.6)

    # PANEL C: ALGORYTMY ZAAWANSOWANE
    ax3.plot(time, nominal, 'k--', alpha=0.2)
    ax3.plot(time, res_nzm, color='purple', linestyle='-', linewidth=1.5, label='Zaawansowany M-z-N')
    ax3.plot(time, res_smoothing, color='orange', linewidth=2, label='Wygładzający (Smoothing)')
    ax3.set_ylabel("Wynik")
    ax3.set_xlabel("Czas [s]")
    ax3.set_title("C. Grupa: Filtry i Algorytmy z Historią", loc='left', fontsize=12, fontweight='bold')
    ax3.legend(loc='upper right', fontsize=9, framealpha=0.9)
    ax3.grid(True, linestyle='--', alpha=0.6)

    file_path = os.path.join(OUTPUT_DIR, f'scenariusz_{scenario}_przebiegi.png')
    plt.savefig(file_path, dpi=200, bbox_inches='tight')
    print(f"Zapisano wykres przebiegów: {file_path}")

    # =========================================================================
    # WYKRES 2: SYGNAŁ BŁĘDU
    # =========================================================================
    plt.figure(figsize=(11, 6))

    plt.plot(time, err_plurality, label='Pluralny', alpha=0.7, linewidth=1)
    plt.plot(time, err_weighted, label='Średnia Ważona', alpha=0.7, linewidth=1)
    plt.plot(time, err_dynamic, 'r--', label='Dynamiczny', linewidth=1.5)
    plt.plot(time, err_nzm, label='Zaawansowany M-z-N', color='purple', alpha=0.8, linewidth=1.5)
    plt.plot(time, err_smooth, label='Wygładzający', color='orange', linewidth=2)

    plt.axhline(0, color='black', linewidth=1.5)
    plt.title(f"Sygnał Błędu (Wzorzec - Wynik Algorytmu)\n{title_text}", fontsize=14)
    plt.xlabel("Czas [s]")
    plt.ylabel("Błąd bezwzględny")
    plt.legend(loc='upper right', ncol=3, fontsize=10)
    plt.grid(True, which='both', linestyle='--', alpha=0.7)

    file_path = os.path.join(OUTPUT_DIR, f'scenariusz_{scenario}_bledy.png')
    plt.savefig(file_path, dpi=200, bbox_inches='tight')
    print(f"Zapisano wykres błędów: {file_path}")

    # =========================================================================
    # WYKRES 3: RANKING MSE
    # =========================================================================
    # Obliczamy MSE
    mse_values = {
        'Pluralny': np.mean(err_plurality ** 2),
        'Średnia Ważona': np.mean(err_weighted ** 2),
        'Dynamiczny': np.mean(err_dynamic ** 2),
        'Zaaw. M-z-N': np.mean(err_nzm ** 2),
        'Wygładzający': np.mean(err_smooth ** 2)
    }

    names = list(mse_values.keys())
    values = list(mse_values.values())

    # Kolorowanie słupków
    best_idx = np.argmin(values)
    colors = ['skyblue'] * len(names)
    colors[best_idx] = 'forestgreen'

    plt.figure(figsize=(10, 6))
    bars = plt.bar(names, values, color=colors, edgecolor='black', alpha=0.8)

    plt.title(f"Ranking Błędu Średniokwadratowego (MSE)\n{title_text}", fontsize=14, fontweight='bold')
    plt.ylabel("MSE (Mniej = Lepiej)")

    # Dodanie wartości nad słupkami
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height, f'{height:.4f}',
                 ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.grid(axis='y', linestyle='--', alpha=0.6)

    file_path = os.path.join(OUTPUT_DIR, f'scenariusz_{scenario}_ranking.png')
    plt.savefig(file_path, dpi=200, bbox_inches='tight')
    print(f"Zapisano ranking MSE: {file_path}")

    print("\nGotowe! Wszystkie pliki zostały wygenerowane w folderze 'wykresy_symulacja_pl'.")