import pandas as pd
from datetime import datetime
import numpy as np

# Lista de archivos de emisiones
archivos = ['Emisioness/emisiones-2016.csv', 'Emisioness/emisiones-2017.csv', 'Emisioness/emisiones-2018.csv', 'Emisioness/emisiones-2019.csv']

# Crear una lista para almacenar los DataFrames
dataframes = []

# Leer cada archivo y añadir el DataFrame a la lista
for archivo in archivos:
    df = pd.read_csv(archivo, sep=';')
    dataframes.append(df)

# Concatenar todos los DataFrames en uno solo
df = pd.concat(dataframes, ignore_index=True)

# Filtrar las columnas requeridas
columnas_deseadas = ['ESTACION', 'MAGNITUD', 'ANO', 'MES'] + [f'D{str(i).zfill(2)}' for i in range(1, 32)]
df = df[columnas_deseadas]

# Reestructurar el DataFrame para los valores de los contaminantes
df_reestructurado = df.melt(id_vars=['ESTACION', 'MAGNITUD', 'ANO', 'MES'], var_name='DIA', value_name='VALOR')

# Añadir una columna de fecha
def construir_fecha(row):
    try:
        año = int(row['ANO'])
        mes = int(row['MES'])
        dia = int(row['DIA'][1:])  # Remover prefijo 'D'
        return datetime(año, mes, dia)
    except ValueError:
        return pd.NaT  # Si no puedes crear una fecha válida, retorna NaT

df_reestructurado['FECHA'] = df_reestructurado.apply(construir_fecha, axis=1)

# Eliminar filas con fechas no válidas (NaT)
df_reestructurado = df_reestructurado[pd.notnull(df_reestructurado['FECHA'])]

# Ordenar el DataFrame
df_ordenado = df_reestructurado.sort_values(by=['ESTACION', 'MAGNITUD', 'FECHA'])

# Mostrar estaciones y contaminantes disponibles
estaciones = df_ordenado['ESTACION'].unique()
contaminantes = df_ordenado['MAGNITUD'].unique()
print("Estaciones disponibles:", estaciones)
print("Contaminantes disponibles:", contaminantes)

# Crear una función para obtener series de emisiones
def obtener_emisiones(estacion, contaminante, fecha_inicio, fecha_fin):
    filtro = (df_ordenado['ESTACION'] == estacion) & (df_ordenado['MAGNITUD'] == contaminante) & (df_ordenado['FECHA'] >= fecha_inicio) & (df_ordenado['FECHA'] <= fecha_fin)
    return df_ordenado[filtro]['VALOR']

# Mostrar resumen descriptivo para cada contaminante
resumen_contaminantes = df_ordenado.groupby('MAGNITUD')['VALOR'].describe()
print("Resumen descriptivo por contaminante:\n", resumen_contaminantes)

# Mostrar resumen descriptivo para cada contaminante por distritos
resumen_contaminantes_estacion = df_ordenado.groupby(['ESTACION', 'MAGNITUD'])['VALOR'].describe()
print("Resumen descriptivo por contaminante y estación:\n", resumen_contaminantes_estacion)

# Función que devuelve resumen descriptivo de emisiones de un contaminante en una estación
def resumen_estacion_contaminante(estacion, contaminante):
    filtro = (df_ordenado['ESTACION'] == estacion) & (df_ordenado['MAGNITUD'] == contaminante)
    return df_ordenado[filtro]['VALOR'].describe()

# Función que devuelve las emisiones medias mensuales de un contaminante y año dado
def emisiones_medias_mensuales(contaminante, año):
    filtro = (df_ordenado['MAGNITUD'] == contaminante) & (df_ordenado['ANO'] == año)
    return df_ordenado[filtro].groupby(['MES', 'ESTACION'])['VALOR'].mean()

# Función que devuelva un DataFrame con las medias mensuales de los distintos tipos de contaminantes
def resumen_mensual_estacion(estacion):
    filtro = (df_ordenado['ESTACION'] == estacion)
    return df_ordenado[filtro].groupby(['MES', 'MAGNITUD'])['VALOR'].mean().unstack()


