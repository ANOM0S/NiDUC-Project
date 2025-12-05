# System tolerujący uszkodzenia czujników lub błędy obliczeniowe z głosowaniem  
Projekt na przedmiot **Niezawodność i diagnostyka układów cyfrowych**

---

## 🎯 Cel projektu

Celem projektu jest opracowanie i przetestowanie systemu redundantnego, w którym kilka równoległych modułów (sensorów) mierzy ten sam sygnał wejściowy.  
Każdy moduł może działać poprawnie lub generować zakłócenia, a system końcowy powinien wybierać za pomocą algorytmów najlepszy wynik.

W projekcie zaimplementowano i porównano dwie metody głosowania z pliku 10a:

- **Algorytm głosowania pluralnego** – wybór największej grupy podobnych wartości  
- **Algorytm głosowania średnia ważona** – średnia ważona z dynamicznie dobieranymi wagami 

I dwie metody z pozostałych plików
- **Algorytm głosowania wygładzający**
- **Algorytm głosowania Wybierający N z M wyników**

---

## 📡 Model sygnału wejściowego

Sygnałem wejściowym do wszystkich sensorów jest **sinusoida**:

\[ 
    x(t) = A \sin(2\pi f t) 
\]

Sensory odczytują ją z niewielkim szumem pomiarowym.

---

## ⚠ Modele awarii sensorów

Każdy sensor posiada prawdopodobieństwo awarii \( p\_fault \).  
W przypadku uszkodzenia zwracany sygnał może zawierać:

### 1. Gaussian Fault  
Sygnał + zakłócenie normalne o dużym odchyleniu:

\[
x_{fault}(t) = x(t) + N(0, \sigma_{fault})
\]

### 2. Outlier Fault  
Losowy impuls zakłócający:

\[
x_{fault}(t) = x(t) + U(-K, K)
\]

### 3. Drift Fault  
Powolne odjeżdżanie amplitudy:

\[
d(t) = d(t-1) + \epsilon, \quad x_{fault}(t) = x(t) + d(t)
\]

### 4. Stuck-at Fault  
Sensor zawiesza się na stałej wartości sygnału.

### 5. Sinusoidal Noise Fault  
Dodatkowy fałszywy sygnał:

\[
x_{fault}(t) = x(t) + B \sin(2\pi f_{noise} t)
\]

---

## 🗳 Zaimplementowane metody głosowania

### ### **1. Plurality Voter**

Metoda tworzy klastry wartości, które są do siebie podobne (różnią się o mniej niż `eps`).

Zwracana jest średnia wartości z największego klastra.

**Zastosowanie:** gdy część sensorów podaje kompletnie błędne wartości (outliery, impulsowe błędy).  
**Zaleta:** prosta i odporna na duże błędy.  
**Wada:** wrażliwa na rozmyte błędy (Gaussian).

---

### **2. Weighted Average Voter**

Każdy sensor dostaje wagę zależną od tego, jak bardzo odstaje od pozostałych:

\[
d_i = \text{średnia odległość sensora } i \text{ od pozostałych}
\]

\[
w_i = \frac{1}{(1 + d_i)^a}
\]

Wynik votera to średnia ważona:

\[
V = \sum_i w_i x_i
\]

**Zaleta:** bardzo dobre wygładzanie danych i odporność na umiarkowane zakłócenia.  
**Wada:** outlier może dostać małą wagę, ale nigdy nie jest ignorowany całkowicie.

---


## 📈 Przykładowe wyniki (opis)

System został przetestowany dla \( N=5 \) sensorów oraz różnych wartości prawdopodobieństwa awarii \( p_{fault} \).

### Oczekiwane zależności:
- **Plurality** radzi sobie lepiej przy impulsowych błędach  
- **Weighted** daje lepszą dokładność przy rozmytych błędach (Gaussian, drift)  
- Obie metody pogarszają się wraz ze wzrostem \( p_{fault} \)

---

## ✔ Wnioski końcowe

- System z redundancją pozwala znacznie zwiększyć odporność na błędy sensorów.  
- Plurality dobrze eliminuje duże odstające błędy.  
- Weighted Average zapewnia lepszą jakość sygnału przy rozmytych szumach i dryfcie.  
- Sinusoidalny sygnał wejściowy pozwala realistyczniej testować zachowanie metod głosowania.  

---

## 📚 Autorzy

Projekt przygotowany jako część laboratorium **NiDUC**.  
Wersja Python przygotowana w oparciu o materiały z wykładów oraz implementację własną.

Miniewski dawid 284556
Osmęda Jan 284691
