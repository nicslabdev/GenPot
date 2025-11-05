import pandas as pd
from io import StringIO

# === Datos de entrada (puedes reemplazar esto por pd.read_csv('archivo.csv')) ===
data = """model,pipeline,network,parse_rate,accuracy,avg_similarity,avg_key_match,n_evaluated
gemma,api,lan,1.0,0.7952380952380952,0.40521469951659383,0.8761904761904762,210
gemma,api,wan,1.0,0.8095238095238095,0.4023069157835687,0.8809523809523809,210
gemma,system,lan,1.0,0.819047619047619,0.4034032238791961,0.8857142857142857,210
gemma,system,wan,1.0,0.8,0.400655599966441,0.8785714285714286,210
llama3,api,lan,1.0,0.6142857142857143,0.35000780911392915,0.7952380952380952,210
llama3,api,wan,1.0,0.5714285714285714,0.35324984405621296,0.7738095238095238,210
llama3,system,lan,1.0,0.5666666666666667,0.3606321155483046,0.7761904761904762,210
llama3,system,wan,1.0,0.5285714285714286,0.34518920879564396,0.7523809523809524,210
zephyr,api,lan,1.0,0.8571428571428571,0.46187574898525474,0.9047619047619048,210
zephyr,api,wan,1.0,0.8571428571428571,0.4527176513068148,0.9047619047619048,210
zephyr,system,lan,1.0,0.8571428571428571,0.4619146998720231,0.9047619047619048,210
zephyr,system,wan,1.0,0.8571428571428571,0.46422569956429516,0.9047619047619048,210
"""

# === Cargar en un DataFrame ===
df = pd.read_csv(StringIO(data))

# === Agrupar por modelo ===
agg_funcs = {
    'parse_rate': 'mean',
    'accuracy': 'mean',
    'avg_similarity': 'mean',
    'avg_key_match': 'mean',
    'n_evaluated': 'sum'
}
df_grouped = df.groupby('model', as_index=False).agg(agg_funcs)

# === Mostrar resultado ===
print("\n📊 Resultados agrupados por modelo:\n")
print(df_grouped.round(3))

# === (Opcional) Exportar a CSV ===
# df_grouped.to_csv('summary_by_model.csv', index=False)


📊 Resultados agrupados por modelo:

    model  parse_rate  accuracy  avg_similarity  avg_key_match  n_evaluated
0   gemma         1.0     0.806           0.403          0.880          840
1  llama3         1.0     0.570           0.352          0.774          840
2  zephyr         1.0     0.857           0.460          0.905          840