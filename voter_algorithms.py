import numpy as np


def formalized_plurality_voter(sensor_outputs, tolerance=0.1):
    """
    Implementacja Formalized Plurality Voter.
    Wybiera wartość, która ma najwięcej sąsiadów w ramach progu tolerancji.
    """
    consensus_value = np.zeros_like(sensor_outputs[0])

    for k in range(len(sensor_outputs[0])):
        # Wyniki ze wszystkich sensorów w danym punkcie czasowym k
        current_readings = sensor_outputs[:, k]
        max_votes = -1
        chosen_value = current_readings[0]  # Domyślna wartość

        # Sprawdzamy każdy odczyt jako potencjalnego kandydata
        for candidate in current_readings:
            # Liczymy "głosy" - ile innych sensorów jest w promieniu tolerancji
            votes = np.sum(np.abs(current_readings - candidate) <= tolerance)

            if votes > max_votes:
                max_votes = votes
                chosen_value = candidate
            elif votes == max_votes:
                # W przypadku remisu można np. wybrać wartość środkową
                pass

        consensus_value[k] = chosen_value

    return consensus_value


def weighted_average_voter(sensor_outputs, weights=None):
    """
    Implementacja Weighted Average Voter.
    Oblicza średnią ważoną wszystkich odczytów.
    """
    num_sensors = sensor_outputs.shape[0]

    # Domyślne wagi - równe dla wszystkich
    if weights is None:
        weights = np.ones(num_sensors) / num_sensors

    if np.sum(weights) != 1.0:
        weights = weights / np.sum(weights)  # Normalizacja wag

    # Obliczanie średniej ważonej dla każdego punktu czasowego
    # sum(W_i * X_i) / sum(W_i) (gdzie W to waga, X to odczyt)
    weighted_sum = np.dot(weights, sensor_outputs)

    return weighted_sum
