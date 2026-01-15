import matplotlib
# Jeśli wykresy nie wyskakują w oknie, odkomentuj linię poniżej (zależnie od systemu):
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
import numpy as np
import sensor_model
import voter_algorithms as va
import analysis_utils as au

# --- KONFIGURACJA ESTETYKI ---
plt.rcParams['font.size'] = 10
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.5
plt.rcParams['lines.linewidth'] = 1.5

if __name__ == '__main__':
    # --- WSPÓLNE PARAMETRY ---
    TIME_POINTS = 300
    AMPLITUDE = 1.0
    BASE_NOISE = 0.05
    TOLERANCE_PLURALITY = 0.3

    # Menu wyboru
    print(
        '''
        DOSTĘPNE SCENARIUSZE TESTOWE:
        -------------------------------------------------------
        1: Baseline (Idealne działanie, tylko szum)
        2: Drift (Jeden sensor powoli odpływa)
        3: Awaria 2 z 3 (Dryft + Szum Gaussowski)
        4: Szpilki/Impulsy (Test odporności na piki)
        5: Chaos (Wysoki szum na wszystkich sensorach)
        6: Heterogeniczny (Awaria głównego sensora)
        '''
    )
    try:
        scenario = int(input("Wybierz scenariusz (1 - 6): "))
    except ValueError:
        scenario = 1

    # --- KONFIGURACJA SCENARIUSZY ---
    match scenario:
        case 1:
            title_text = "Baseline (Szum bazowy)"
            NUM_SENSORS = 3
            STATIC_WEIGHTS = [1 / 3, 1 / 3, 1 / 3]
            DRIFTING_SENSORS = []
            test_scenarios = []
        case 2:
            title_text = "Awaria Pojedyncza (Drift)"
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
            title_text = "Zakłócenia Impulsowe (Szpilki)"
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
            STATIC_WEIGHTS = [0.6, 0.2, 0.2]
            DRIFTING_SENSORS = [0]
            test_scenarios = [
                {'sensor_index': 0, 'fault_type': 'drift', 'params': {'drift_rate': 0.05}},
                {'sensor_index': 1, 'fault_type': 'gaussian', 'params': {'mean': 0.0, 'std': 0.1}},
                {'sensor_index': 2, 'fault_type': 'gaussian', 'params': {'mean': 0.0, 'std': 0.1}}
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

    res_nzm = []
    res_smoothing = []
    res_dynamic_weighted = []
    prev_val_smooth = None

    for t_idx in range(TIME_POINTS):
        readings = data[:, t_idx]
        res_nzm.append(va.n_z_m_voter(readings, m_to_keep=2))

        val_smooth = va.smoothing_voter(readings, prev_val_smooth, alpha=0.15)
        prev_val_smooth = val_smooth
        res_smoothing.append(val_smooth)

        res_dynamic_weighted.append(va.weighted_average_dynamic_bron(readings, a=0.5))

    res_nzm = np.array(res_nzm)
    res_smoothing = np.array(res_smoothing)
    res_dynamic_weighted = np.array(res_dynamic_weighted)

    # --- ANALIZA NUMERYCZNA ---
    au.display_numerical_results(nominal, data, res_plurality, res_weighted_static, res_nzm, res_smoothing,
                                 res_dynamic_weighted, time)
    au.calculate_and_display_mse(nominal, res_plurality, res_weighted_static, res_nzm, res_smoothing,
                                 res_dynamic_weighted)

    # =========================================================================
    # WYKRES 1: KOMPLEKSOWA ANALIZA CZASOWA (3 PANELE)
    # =========================================================================
    # Używamy mniejszej wysokości (8.5), żeby mieściło się na ekranie
    fig1, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(11, 8.5), sharex=True)

    fig1.subplots_adjust(top=0.92, hspace=0.3)
    fig1.suptitle(f"Scenariusz: {title_text}", fontsize=14, fontweight='bold')

    # --- PANEL A: SENSORY ---
    ax1.plot(time, nominal, 'k--', linewidth=1.5, label='Wzorzec')
    for i in range(NUM_SENSORS):
        label_text = f'S{i}'
        if i in DRIFTING_SENSORS:
            ax1.plot(time, data[i, :], color='red', alpha=0.6, linestyle=':', linewidth=2,
                     label=label_text + ' (Awaria)')
        else:
            ax1.plot(time, data[i, :], color='gray', alpha=0.4, linewidth=1, label=label_text)
    ax1.set_ylabel("Amplituda")
    ax1.set_title("A. Dane z sensorów", loc='left', fontsize=10, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=8, framealpha=0.9)

    # --- PANEL B: ALGORYTMY GŁOSUJĄCE (Plurality + Weighted) ---
    ax2.plot(time, nominal, 'k--', alpha=0.2)
    ax2.plot(time, res_plurality, color='blue', linewidth=1.2, label='Plurality')
    ax2.plot(time, res_weighted_static, color='green', linewidth=1.2, label='Weighted (Static)')
    ax2.plot(time, res_dynamic_weighted, color='crimson', linestyle='--', linewidth=1.5, label='Dynamic (Brøn)')

    ax2.set_ylabel("Wynik")
    ax2.set_title("B. Grupa: Średnie i Głosowanie", loc='left', fontsize=10, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=8, framealpha=0.9)

    # --- PANEL C: ALGORYTMY FILTRUJĄCE (N-z-M + Smoothing) ---
    ax3.plot(time, nominal, 'k--', alpha=0.2)
    ax3.plot(time, res_nzm, color='purple', linestyle='-', linewidth=1.2, label='N-z-M')
    ax3.plot(time, res_smoothing, color='orange', linewidth=2, label='Smoothing')

    ax3.set_ylabel("Wynik")
    ax3.set_xlabel("Czas")
    ax3.set_title("C. Grupa: Filtry i Odrzucanie Skrajnych", loc='left', fontsize=10, fontweight='bold')
    ax3.legend(loc='upper right', fontsize=8, framealpha=0.9)

    plt.savefig(f'scenariusz_{scenario}_przebiegi.png', dpi=150)
    plt.show()

    # =========================================================================
    # WYKRES 2: ANALIZA BŁĘDU (Error Plot)
    # =========================================================================
    plt.figure(figsize=(11, 5))

    err_plurality = nominal - res_plurality
    err_weighted = nominal - res_weighted_static
    err_nzm = nominal - res_nzm
    err_smooth = nominal - res_smoothing
    err_dynamic = nominal - res_dynamic_weighted

    plt.plot(time, err_plurality, label='Plurality', alpha=0.6, linewidth=1)
    plt.plot(time, err_weighted, label='Weighted', alpha=0.6, linewidth=1)
    plt.plot(time, err_dynamic, 'r--', label='Dynamic', linewidth=1.5)
    plt.plot(time, err_nzm, label='N-z-M', alpha=0.6, linewidth=1)
    plt.plot(time, err_smooth, label='Smoothing', linewidth=2)

    plt.axhline(0, color='black', linewidth=1.5)
    plt.title(f"Sygnał Błędu (Nominalny - Wyjście Votera)\n{title_text}")
    plt.xlabel("Czas")
    plt.ylabel("Błąd")
    plt.legend(loc='upper right', ncol=3, fontsize=8)
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f'scenariusz_{scenario}_bledy.png', dpi=150)
    plt.show()

    # =========================================================================
    # WYKRES 3: RANKING MSE (Bar Chart)
    # =========================================================================
    mse_values = {
        'Plurality': np.mean(err_plurality ** 2),
        'Weighted': np.mean(err_weighted ** 2),
        'Dynamic': np.mean(err_dynamic ** 2),
        'N-z-M': np.mean(err_nzm ** 2),
        'Smoothing': np.mean(err_smooth ** 2)
    }

    names = list(mse_values.keys())
    values = list(mse_values.values())
    best_idx = np.argmin(values)
    colors = ['skyblue'] * len(names)
    colors[best_idx] = 'forestgreen'

    plt.figure(figsize=(10, 5))
    bars = plt.bar(names, values, color=colors, edgecolor='black', alpha=0.8)
    plt.title(f"Ranking MSE (Błąd Średniokwadratowy) - {title_text}", fontweight='bold')
    plt.ylabel("MSE (Mniej = Lepiej)")

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height, f'{height:.4f}', ha='center', va='bottom')

    plt.grid(axis='y', linestyle='--')
    plt.tight_layout()
    plt.savefig(f'scenariusz_{scenario}_ranking.png', dpi=150)
    plt.show()

    print("\nGotowe! Wygenerowano wykresy i zapisano jako pliki PNG.")