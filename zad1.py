import math

# Obliczenie wartości
obliczenie = (math.sqrt(32) - math.sqrt(2))**2

# Ze względu na specyfikę liczb zmiennoprzecinkowych w Pythonie, 
# wynik wynosi około 17.999999999999996. Musimy go zaokrąglić.
wynik = round(obliczenie)

print(f"Wynik obliczeń: {wynik}")