import pandas as pd
import numpy as np

# Lista de archivos de emisiones
archivos_df= ['Emisioness/emisiones-2016.csv', 'Emisioness/emisiones-2017.csv', 'Emisioness/emisiones-2018.csv', 'Emisioness/emisiones-2019.csv']

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



