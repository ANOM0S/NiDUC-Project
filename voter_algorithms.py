import numpy as np


def formalized_plurality_voter(sensor_outputs, tolerance=0.1):
    """
    1. Formalized Plurality Voter.
    Wybiera wartość, która ma najwięcej sąsiadów w ramach progu tolerancji.
    """
    consensus_value = np.zeros_like(sensor_outputs[0])

    for k in range(len(sensor_outputs[0])):
        current_readings = sensor_outputs[:, k]
        max_votes = -1
        chosen_value = current_readings[0]

        for candidate in current_readings:
            votes = np.sum(np.abs(current_readings - candidate) <= tolerance)

            if votes > max_votes:
                max_votes = votes
                chosen_value = candidate
            # W razie remisu zostaje pierwszy lepszy (uproszczenie)

        consensus_value[k] = chosen_value

    return consensus_value


def weighted_average_voter(sensor_outputs, weights=None):
    """
    2. Weighted Average Voter (Static).
    Oblicza średnią ważoną ze stałymi wagami.
    """
    num_sensors = sensor_outputs.shape[0]
    if weights is None:
        weights = np.ones(num_sensors) / num_sensors

    # Upewniamy się, że wagi to numpy array
    weights = np.array(weights)

    if np.sum(weights) != 1.0:
        weights = weights / np.sum(weights)

    # Iloczyn skalarny wag i odczytów
    weighted_sum = np.dot(weights, sensor_outputs)
    return weighted_sum


def weighted_average_dynamic_bron(sensor_values, a=10.0):
    """
    3. Dynamic Weighted Voter (Brøn).
    Liczy wagi dynamicznie na podstawie odległości sensorów od siebie.
    """
    x = np.array(sensor_values, dtype=float)
    N = len(x)

    # Suma odległości dla każdego sensora od reszty
    S = np.zeros(N)
    for i in range(N):
        for j in range(N):
            if i != j:
                S[i] += abs(x[i] - x[j])

    # Wagi surowe (odwrotność odległości)
    # Dodajemy 'a' w mianowniku, żeby nie dzielić przez zero
    w_raw = 1.0 / (a + S)

    # Normalizacja wag do jedynki
    if np.sum(w_raw) == 0:
        w = np.ones(N) / N
    else:
        w = w_raw / np.sum(w_raw)

    return np.sum(w * x)


def n_z_m_voter(sensor_values, m_to_keep=2):
    """
    4. N-z-M Voter (Trimmed Mean).
    Odrzuca skrajne wartości i liczy średnią z M środkowych.
    """
    n = len(sensor_values)
    if m_to_keep >= n:
        return np.mean(sensor_values)

    sorted_vals = np.sort(sensor_values)

    # Ile odrzucić łącznie
    to_remove = n - m_to_keep
    # Ile z dołu, ile z góry
    cut_low = to_remove // 2
    cut_high = n - (to_remove - cut_low)

    # Wybieramy środek
    selected = sorted_vals[cut_low:cut_high]

    return np.mean(selected)


def smoothing_voter(sensor_values, previous_result, alpha=0.25):
    """
    5. Smoothing Voter (Wygładzający).
    Łączy obecną medianę z poprzednim wynikiem (pamięć).
    """
    current_estimate = np.median(sensor_values)

    if previous_result is None:
        return current_estimate

    # Filtr dolnoprzepustowy (Exponential Smoothing)
    result = alpha * current_estimate + (1 - alpha) * previous_result
    return result