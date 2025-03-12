import os
import time

def get_cpu_temp():
    """Odczytuje temperaturę CPU z systemu Raspberry Pi."""
    res = os.popen('vcgencmd measure_temp').readline()
    return res.replace("temp=", "").replace("'C\n", "")


# Plik, do którego będziemy zapisywać dane
log_file_path = "cpu_log.txt"

while True:
    cpu_temp = get_cpu_temp()
    with open(log_file_path, "a") as log_file:  # Otwieranie pliku w trybie dopisywania
        log_file.write(f"Temperatura CPU: {cpu_temp} °C%\n")
    
    print(f"Temperatura CPU: {cpu_temp} °C")
    time.sleep(5)  # Częstotliwość aktualizacji co 5 sekund
