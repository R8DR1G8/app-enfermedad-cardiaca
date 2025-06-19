import pandas as pd

# Leer el archivo Excel
df = pd.read_excel("HeartDisease_Formatted.xlsx")

# Guardarlo como CSV
df.to_csv("HeartDisease_Formatted.csv", index=False)

print("✅ Excel convertido correctamente a CSV")
