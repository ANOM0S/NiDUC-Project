import numpy as np

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