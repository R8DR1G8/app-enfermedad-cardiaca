import matplotlib.pyplot as plt
import pandas as pd

# Datos exactos de tu Excel
data = {
    'Año': [1990, 1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999, 2000, 
            2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 
            2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024],
    'Emigrantes': [46596, 68349, 58789, 78944, 38039, 26626, 27253, 32499, 40998, 34917, 
                   40058, 41608, 62941, 84921, 114877, 141490, 194136, 202734, 199777, 
                   198231, 180725, 169078, 163822, 147564, 141688, 136921, 120683, 
                   107155, 119318, 120982, 98308, 127564, 260623, 310686, 605000]
}

df = pd.DataFrame(data)

# Configuración profesional de la gráfica
plt.figure(figsize=(12, 6))
plt.plot(df['Año'], df['Emigrantes'], marker='o', color='#1f77b4', linestyle='-', linewidth=2)
plt.title('Evolución del Número de Emigrantes (1990-2024)', fontsize=14, pad=20)
plt.xlabel('Año', fontsize=12, labelpad=10)
plt.ylabel('Número de Emigrantes', fontsize=12, labelpad=10)
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(range(1990, 2025, 2), rotation=45)
plt.tight_layout()

# Mostrar gráfica
plt.show()