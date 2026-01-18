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


def advanced_m_out_of_n_voter(readings, weights, threshold_m, threshold_tau, threshold_gamma, previous_result=None):
    """
    Implementacja Advanced M-out-of-N Voting Algorithm na podstawie:
    "A Novel N-Input Voting Algorithm for X-by-Wire Fault-Tolerant Systems" (Algorithm 1).

    Parametry:
    - readings: lista odczytów z sensorów [x1, x2, ...]
    - weights: wagi sensorów [v1, v2, ...]
    - threshold_m: próg sumy wag wymagany do akceptacji (M)
    - threshold_tau: próg zgodności (tolerancja) do grupowania (tau)
    - threshold_gamma: próg ciągłości względem poprzedniego wyniku (gamma)
    - previous_result: wynik systemu z poprzedniego cyklu (dla Fazy 2)
    """
    n = len(readings)

    # --- FAZA 1: Głosowanie w grupach (Exact/Inexact Voting) ---
    # Algorytm z papieru używa "slotów". Dla N=3 uprościmy to do bezpośredniego grupowania.
    # Każdy 'slot' to reprezentant grupy (object_j) i suma wag (tally_j).

    slots_objects = []  # Reprezentanci grup
    slots_tallies = []  # Sumy wag w grupach

    # Krok 1: Budowanie grup (uproszczona wersja linii 1-15 dla małego N)
    for i in range(n):
        x_i = readings[i]
        v_i = weights[i]

        found_group = False
        for j in range(len(slots_objects)):
            # Sprawdzamy czy x_i pasuje do istniejącej grupy (odległość <= tau)
            if abs(x_i - slots_objects[j]) <= threshold_tau:
                slots_tallies[j] += v_i
                found_group = True
                break

        if not found_group:
            # Tworzymy nową grupę
            slots_objects.append(x_i)
            slots_tallies.append(v_i)

    # Krok 2: Sprawdzenie progu M (linie 16-19)
    # Szukamy grupy, która ma sumę wag >= M
    for j in range(len(slots_objects)):
        if slots_tallies[j] >= threshold_m:
            return slots_objects[j]  # Sukces w fazie 1: Zwracamy reprezentanta grupy

    # --- FAZA 2: Weryfikacja historyczna (linie 21-26) ---
    # Jeśli nie ma zgody w Fazie 1, sprawdzamy spójność z historią.

    if previous_result is None:
        return 0.0  # Brak historii i brak zgody -> Benign output / Fail-safe (tutaj 0.0)

    # Obliczamy odległości wszystkich wejść od poprzedniego wyniku
    distances = [abs(x - previous_result) for x in readings]

    # Znajdujemy minimalną odległość
    min_dist = min(distances)
    min_index = distances.index(min_dist)

    # Jeśli najbliższy sensor jest w granicach gammy, ufamy mu
    if min_dist <= threshold_gamma:
        return readings[min_index]

    # --- FAZA 3: Brak rozstrzygnięcia ---
    # Zgodnie z papierem: "voter fails to produce output... benign output"
    # W symulacji najlepiej zwrócić poprzednią wartość (hold) lub 0.
    return previous_result


import numpy as np


# --- POMOCNICZA FUNKCJA DO SZUKANIA GRUP (używana w Plurality i Smoothing) ---
def find_majority_group(readings, tolerance):
    """
    Pomocnicza funkcja szukająca grupy większościowej (Majority/Consensus).
    Zwraca (Success, Value). Success = True jeśli znaleziono grupę > N/2.
    """
    n = len(readings)
    # Proste grupowanie (jak w Plurality)
    for i in range(n):
        group_indices = [i]
        for j in range(n):
            if i == j: continue
            if abs(readings[i] - readings[j]) <= tolerance:
                group_indices.append(j)

        # Sprawdzenie warunku większości (zgodnie z art. Majority > N/2)
        # Dla N=3 potrzeba 2 zgodnych.
        if len(group_indices) > n / 2:
            # Zwracamy średnią z grupy jako wynik
            group_vals = [readings[k] for k in group_indices]
            return True, np.mean(group_vals)

    return False, 0.0


def smoothing_voter(readings, previous_result, threshold_majority_epsilon, threshold_smoothing_beta):
    """
    Implementacja Smoothing Voter zgodnie z artykułem:
    "Smoothing voter: a novel voting algorithm..." (Latif-Shabgahi et al., 2003).

    Zasada (sekcja 3.1 artykułu):
    1. Sprawdź, czy istnieje większość (Majority) z progiem epsilon.
    2. Jeśli TAK -> Zwróć wynik większości.
    3. Jeśli NIE (Complete Disagreement) -> Znajdź wynik najbliższy previous_result.
    4. Jeśli odległość <= beta -> Zwróć ten wynik.
    5. W przeciwnym razie -> Brak wyniku (zwracamy previous lub 0).
    """

    # KROK 1: Sprawdzenie Większości (Steps S3-S4 w artykule)
    has_majority, majority_val = find_majority_group(readings, threshold_majority_epsilon)

    if has_majority:
        return majority_val  # Priorytet ma zawsze zgoda sensorów!

    # KROK 2: Wygładzanie historyczne (Step S5-S6 w artykule)
    # Uruchamiane TYLKO gdy nie ma zgody między sensorami.

    if previous_result is None:
        # Jeśli to pierwszy krok i brak zgody -> np. średnia (lub 0)
        return np.mean(readings)

    # Szukamy sensora najbliższego historii
    distances = [abs(x - previous_result) for x in readings]
    min_dist = min(distances)
    min_index = distances.index(min_dist)
    candidate_value = readings[min_index]

    # Sprawdzenie progu wygładzania (Beta)
    if min_dist <= threshold_smoothing_beta:
        return candidate_value
    else:
        # Voter fails to produce output (w artykule: "no result").
        # W symulacji utrzymujemy poprzednią wartość (fail-safe hold).
        return previous_result