# Algorytmy Głosowania w Systemach Redundantnych (NiDUC)

## 🚀 Wprowadzenie

Ten projekt stanowi symulację systemów redundantnych z sensorami, mającą na celu analizę i porównanie skuteczności  algorytmów głosowania: **Formalized Plurality Voter** oraz **Weighted Average Voter** .

Symulacja generuje sygnał nominalny (sinusoidalny) oraz jego zaszumione kopie (dane z sensorów), wstrzykując różne typy awarii (dryft, *stuck-at-value*, szpilki), aby ocenić niezawodność każdego algorytmu.

## 🛠️ Wymagania i Instalacja

Do uruchomienia skryptu wymagane są następujące biblioteki Python:

```bash
numpy
matplotlib
pandas
scikit-learn
tabulate

