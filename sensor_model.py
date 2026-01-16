import numpy as np


def generate_sensor_data(time_points, num_sensors, amplitude=1.0, frequency=1.0,
                         base_noise_level=0.05, fault_scenarios=None):
    t = np.linspace(0, 4 * np.pi, time_points)
    nominal_signal = amplitude * np.sin(frequency * t)
    sensor_data = np.zeros((num_sensors, time_points))

    for i in range(num_sensors):
        current_signal = np.copy(nominal_signal)
        current_signal += np.random.normal(0, base_noise_level, time_points)

        if fault_scenarios:
            for fault in fault_scenarios:
                if fault['sensor_index'] == i:
                    f_type = fault['fault_type']
                    params = fault['params']

                    if f_type == 'drift':
                        drift_rate = params.get('drift_rate', 0.01)
                        current_signal += np.linspace(0, drift_rate * time_points, time_points)

                    elif f_type == 'gaussian':
                        current_signal += np.random.normal(params.get('mean', 0), params.get('std', 0.5), time_points)

                    elif f_type == 'outlier':
                        prob = params.get('prob', 0.05)
                        mask = np.random.rand(time_points) < prob
                        current_signal[mask] += np.random.uniform(params.get('min_val', -2), params.get('max_val', 2),
                                                                  np.sum(mask))
                    elif f_type == 'stuck':
                        # Sensor zacina się na stałej wartości od pewnego momentu lub całościowo
                        val = params.get('value', 0.0)  # Na jakiej wartości się zaciął
                        start_idx = params.get('start_idx', 0)  # Od kiedy (0 = od początku)
                        current_signal[start_idx:] = val

                    elif f_type == 'bias':
                        # Stałe przesunięcie (błąd kalibracji)
                        offset = params.get('offset', 0.5)
                        current_signal += offset

        sensor_data[i, :] = current_signal

    return nominal_signal, sensor_data, t