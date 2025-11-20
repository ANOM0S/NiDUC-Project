import numpy as np


class UniversalSensor:
    def __init__(self, true_value, p_fault=0.1, fault_type="gaussian", metric="auto", sigma_good=0.01, sigma_bad=1.0,
                 modulo_base=None):
        """
        true_value  – prawdziwa wartość (może być: liczba, wektor, lista bitów, string)
        p_fault     – prawdopodobieństwo awarii
        fault_type  – rodzaj błędu: gaussian, drift, stuck, outlier, bitflip
        metric      – metryka: auto, euclidean, manhattan, hamming, modulo
        modulo_base – baza do liczenia odległości modulo (np. 32768)
        """
        self.true_value = true_value
        self.p_fault = p_fault
        self.fault_type = fault_type
        self.metric = metric
        self.sigma_good = sigma_good
        self.sigma_bad = sigma_bad

        self.drift_offset = 0
        self.stuck_value = None
        self.modulo_base = modulo_base

    # --------------------
    #      MODELE BŁĘDÓW
    # --------------------

    def noise_gaussian(self):
        return self.true_value + np.random.normal(0, self.sigma_bad)

    def noise_stuck(self):
        if self.stuck_value is None:
            # losowa stała wartość odsunięta od true_value
            self.stuck_value = self.true_value + np.random.uniform(-5, 5)
        return self.stuck_value

    def noise_drift(self):
        self.drift_offset += np.random.normal(0, 0.05)
        return self._add_to_value(self.drift_offset)

    def noise_outlier(self):
        # losowa wartość z dużego zakresu
        if isinstance(self.true_value, (list, np.ndarray)):
            return np.random.uniform(-100, 100, size=len(self.true_value))
        return np.random.uniform(-100, 100)

    def noise_bitflip(self):
        if isinstance(self.true_value, (int, np.integer)):
            return 1 - self.true_value
        if isinstance(self.true_value, (list, np.ndarray)):
            return np.array([1 - int(x) for x in self.true_value])
        raise Exception("Bitflip można stosować tylko do wartości binarnych.")

    # --------------------
    #    FUNKCJE POMOCNICZE
    # --------------------

    def _add_to_value(self, noise):
        """
        Dodaje szum do wartości, wspierając wektory i liczby.
        """
        if isinstance(self.true_value, (list, np.ndarray)):
            return np.array(self.true_value) + noise
        else:
            return self.true_value + noise

    # --------------------
    #       ODCZYT
    # --------------------

    def read(self):
        # brak awarii
        if np.random.rand() > self.p_fault:
            return self._add_to_value(np.random.normal(0, self.sigma_good))

        # AWARIA
        if self.fault_type == "gaussian":
            return self.noise_gaussian()
        elif self.fault_type == "stuck":
            return self.noise_stuck()
        elif self.fault_type == "drift":
            return self.noise_drift()
        elif self.fault_type == "outlier":
            return self.noise_outlier()
        elif self.fault_type == "bitflip":
            return self.noise_bitflip()
        else:
            return self.noise_gaussian()  # fallback

    # --------------------
    #       METRYKI
    # --------------------

    def distance(self, a, b):
        """
        Automatyczna metryka:
        - liczby → |a - b|
        - wektory → euclidean
        - bity → Hamming
        - modulo → metryka okręgowa
        """
        # wymuszenie określonej metryki przez użytkownika
        if self.metric == "euclidean":
            return np.linalg.norm(np.array(a) - np.array(b))
        if self.metric == "manhattan":
            return np.sum(np.abs(np.array(a) - np.array(b)))
        if self.metric == "hamming":
            return np.sum(np.array(a) != np.array(b))
        if self.metric == "modulo":
            return min(abs(a - b), self.modulo_base - abs(a - b))

        # tryb AUTO — sam wybiera metrykę

        # 1. wartości skalarne (int/float)
        if isinstance(a, (int, float, np.integer, np.floating)):
            if self.modulo_base is not None:
                return min(abs(a - b), self.modulo_base - abs(a - b))
            return abs(a - b)

        # 2. wektory lub listy liczb
        if isinstance(a, (list, np.ndarray)):
            arr_a, arr_b = np.array(a), np.array(b)
            # dla binarnych wektorów — Hamming
            if arr_a.dtype == int and np.all((arr_a == 0) | (arr_a == 1)):
                return np.sum(arr_a != arr_b)
            # domyślnie euclidean
            return np.linalg.norm(arr_a - arr_b)

        # 3. kategorie (stringi)
        if isinstance(a, str):
            return 0 if a == b else 1  # najprostsza metryka

        raise Exception(f"Nieznany typ danych dla distance(): {type(a)}")


s = UniversalSensor(10.0, fault_type="gaussian")
print(s.read())

s = UniversalSensor([1, 2, 3], fault_type="drift")
print(s.read())

s = UniversalSensor([1, 0, 1, 1], fault_type="bitflip", metric="hamming")
print(s.read())

s = UniversalSensor(30000, fault_type="gaussian", modulo_base=32768)
print(s.distance(30000, 100))  # poprawnie policzy modulo
