import numpy as np


def generate_sensor_data(time_points, num_sensors, amplitude=1.0, frequency=1.0,
                         base_noise_level=0.1, fault_scenarios=None):
    """
    Generuje nominalny sygnał sinusoidalny i zaszumione dane z sensorów
    z możliwością symulowania różnych scenariuszy awarii.

    :param base_noise_level: Poziom szumu podstawowego (losowego, normalnego).
    :param fault_scenarios: Lista słowników definiujących awarie.
        Typy: 'bias', 'spikes', 'stuck', 'freeze'.
    """

    t = np.linspace(0, 2 * np.pi, time_points)
    nominal_signal = amplitude * np.sin(frequency * t)
    sensor_data = np.zeros((num_sensors, time_points))

    for i in range(num_sensors):

        current_sensor_signal = np.copy(nominal_signal)

        # 1. Podstawowe zakłócenie (szum normalny dla wszystkich)
        base_noise = np.random.normal(0, base_noise_level, time_points)
        current_sensor_signal += base_noise

        # 2. Awarie specyficzne dla sensora (Fault Injection)
        if fault_scenarios:
            for fault in fault_scenarios:
                if i in fault.get('sensors', []):

                    fault_type = fault['type']
                    magnitude = fault.get('magnitude', 1.0)

                    if fault_type == 'bias':
                        # AWARIA: Dryft / Stałe przesunięcie
                        current_sensor_signal += magnitude

                    elif fault_type == 'stuck':
                        # AWARIA: Stuck-At-Value (Sensor zwraca stałą wartość)
                        stuck_value = fault.get('value', 0.0)
                        current_sensor_signal[:] = stuck_value

                    elif fault_type == 'spikes':
                        # AWARIA: Szpilki (impulsowe zakłócenia)
                        density = fault.get('density', 0.01)
                        num_spikes = int(time_points * density)
                        spike_indices = np.random.choice(time_points, size=num_spikes, replace=False)
                        spike_values = np.random.uniform(-magnitude, magnitude, num_spikes)
                        spike_mask = np.zeros(time_points)
                        spike_mask[spike_indices] = spike_values
                        current_sensor_signal += spike_mask

                    elif fault_type == 'freeze':
                        # AWARIA: Freeze Fault (trzymanie wartości z momentu awarii)
                        freeze_start_time = fault.get('time_point', time_points // 4)

                        if freeze_start_time < time_points:
                            freeze_value = current_sensor_signal[freeze_start_time]
                            current_sensor_signal[freeze_start_time:] = freeze_value

        sensor_data[i, :] = current_sensor_signal

    return nominal_signal, sensor_data, t