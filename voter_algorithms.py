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


def weighted_average_dynamic_bron(sensor_values, a=10.0):
    """
    Dynamic Weighted Average Voter based on Brøn (1975)

    :param sensor_values: list or np.array of sensor outputs [x1, x2, ..., xN]
    :param a: scaling constant
    :return: voted output (float)
    """
    x = np.array(sensor_values, dtype=float)
    N = len(x)

    # Sum of distances for each sensor
    S = np.zeros(N)
    for i in range(N):
        for j in range(N):
            if i != j:
                S[i] += abs(x[i] - x[j])

    # Raw weights (inverse of distance sum)
    w_raw = 1.0 / (a + S)

    # Normalize weights
    w = w_raw / np.sum(w_raw)

    # Weighted average output
    return np.sum(w * x)


def n_z_m_voter(sensor_values, m_to_keep):
    """
    Algorytm N z M (Trimmed Mean).
    Sortuje wartości sensorów, odrzuca skrajne i liczy średnią z 'm_to_keep' środkowych wyników.
    Dla 3 sensorów i m=2 odrzuci jedną najbardziej skrajną wartość (lub dwie skrajne symetrycznie, zależnie od logiki).
    Tutaj wersja symetryczna: odcina tyle samo z góry i dołu (lub niesymetrycznie o 1, jeśli różnica jest nieparzysta).
    """
    n = len(sensor_values)
    if m_to_keep >= n:
        return np.mean(sensor_values)

    # Sortujemy odczyty
    sorted_vals = np.sort(sensor_values)

    # Obliczamy ile odrzucić
    to_remove = n - m_to_keep
    cut_low = to_remove // 2
    cut_high = n - (to_remove - cut_low)

    # Wybieramy środek
    selected = sorted_vals[cut_low:cut_high]

    return np.mean(selected)


def smoothing_voter(sensor_values, previous_result, alpha=0.7):
    """
    Algorytm Wygładzający.
    Łączy obecną estymatę (np. medianę z sensorów) z wynikiem z poprzedniej iteracji.
    alpha: waga obecnego pomiaru (0.0 - 1.0). Mniejsze alpha = mocniejsze wygładzanie (wolniejsza reakcja).
    """
    # Krok 1: Wstępna ocena wartości w tej chwili (np. mediana jest odporna na pojedyncze błędy)
    current_estimate = np.median(sensor_values)

    # Krok 2: Jeśli nie ma historii (pierwszy krok), zwróć obecną
    if previous_result is None:
        return current_estimate

    # Krok 3: Średnia ważona z historią (filtr dolnoprzepustowy)
    result = alpha * current_estimate + (1 - alpha) * previous_result
    return result