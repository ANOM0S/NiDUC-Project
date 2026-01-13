import numpy as np


def generate_sensor_data(time_points, num_sensors, amplitude=1.0, frequency=1.0,
                         base_noise_level=0.05, fault_scenarios=None):
    """
    Generuje sygnał i aplikuje awarie zgodnie z definicjami w main.py
    (Drift, Gaussian, Outlier).
    """
    t = np.linspace(0, 4 * np.pi, time_points)  # Wydłużyłem czas, żeby było widać więcej cykli
    nominal_signal = amplitude * np.sin(frequency * t)
    sensor_data = np.zeros((num_sensors, time_points))

    # Dla każdego sensora generujemy bazowy sygnał z szumem
    for i in range(num_sensors):
        # Kopia idealnego sygnału
        current_signal = np.copy(nominal_signal)

        # Dodajemy bazowy szum pomiarowy (mały, występujący zawsze)
        current_signal += np.random.normal(0, base_noise_level, time_points)

        # --- APLIKOWANIE AWARII (FAULT INJECTION) ---
        if fault_scenarios:
            for fault in fault_scenarios:
                # Sprawdzamy, czy ta awaria dotyczy tego sensora
                if fault['sensor_index'] == i:
                    f_type = fault['fault_type']
                    params = fault['params']

                    if f_type == 'drift':
                        # Symulacja narastającego błędu (offset rośnie w czasie)
                        drift_rate = params.get('drift_rate', 0.01)
                        drift_vector = np.linspace(0, drift_rate * time_points, time_points)
                        current_signal += drift_vector

                    elif f_type == 'gaussian':
                        # Dodatkowy, silny szum (np. uszkodzona elektronika)
                        mean = params.get('mean', 0.0)
                        std = params.get('std', 0.5)
                        noise = np.random.normal(mean, std, time_points)
                        current_signal += noise

                    elif f_type == 'outlier':
                        # Losowe szpilki (impulsy)
                        prob = params.get('prob', 0.05)
                        min_val = params.get('min_val', -2.0)
                        max_val = params.get('max_val', 2.0)

                        # Maska gdzie wystąpią szpilki
                        outlier_mask = np.random.rand(time_points) < prob
                        # Wartości szpilek
                        outliers = np.random.uniform(min_val, max_val, time_points)

                        # Nadpisujemy sygnał w miejscach wystąpienia szpilek (lub dodajemy)
                        # Tutaj dodajemy do sygnału:
                        current_signal[outlier_mask] += outliers[outlier_mask]

        sensor_data[i, :] = current_signal

    return nominal_signal, sensor_data, t