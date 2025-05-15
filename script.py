import pandas as pd
df = pd.read_csv('titanic.csv')

# 2. Mostrarremos en pantalla las dimensiones del DataFrame, el número de datos que contiene,
# los nombres de sus columnas y filas, los tipos de datos de las columnas,
# las 10 primeras filas y las 10 últimas filas.
#Cabe aclarar que me gustó la idea de ordenarlos así por tema estético.
print("Dimensiones del DataFrame:", df.shape)
print("Número de datos:", df.size)
print("Nombres de las columnas:", df.columns)
print("Tipos de datos de las columnas:\n", df.dtypes)
print("Primeras 10 filas:\n", df.head(10))
print("Últimas 10 filas:\n", df.tail(10))

# 3. Mostraremos los datos del pasajero con identificador 148.
print("Datos del pasajero con ID 148:\n", df[df['PassengerId'] == 148])

# 4. Mostraremos las filas pares del DataFrame.
print("Filas pares del DataFrame:\n", df.iloc[::2])

# 5. Mostrar los nombres de las personas que iban en primera clase ordenadas alfabéticamente.
first_class_names = df[df['Pclass'] == 1]['Name'].sort_values()
print("Nombres de personas en primera clase ordenados alfabéticamente:\n", first_class_names)

# 6. Mostraremos el porcentaje de personas que sobrevivieron y murieron.
survival_rate = df['Survived'].value_counts(normalize=True) * 100
print("Porcentaje de personas que sobrevivieron y murieron:\n", survival_rate)

# 7. Mostraremos el porcentaje de personas que sobrevivieron en cada clase.
survival_rate_class = df.groupby('Pclass')['Survived'].mean() * 100
print("Porcentaje de supervivencia por clase:\n", survival_rate_class)

# 8. Eliminar del DataFrame los pasajeros con edad desconocida.
df = df.dropna(subset=['Age'])

# 9. Mostraremos la edad media de las mujeres que viajaban en cada clase.
average_age_women_by_class = df[df['Sex'] == 'female'].groupby('Pclass')['Age'].mean()
print("Edad media de las mujeres por clase:\n", average_age_women_by_class)

# 10. Añadir una nueva columna booleana para ver si el pasajero era menor de edad o no.
df['IsMinor'] = df['Age'] < 18

# 11. Mostraremos el porcentaje de menores y mayores de edad que sobrevivieron en cada clase.
survival_rate_minors_class = df[df['IsMinor']].groupby('Pclass')['Survived'].mean() * 100
survival_rate_adults_class = df[~df['IsMinor']].groupby('Pclass')['Survived'].mean() * 100
print("Porcentaje de supervivencia de menores de edad por clase:\n", survival_rate_minors_class)
print("Porcentaje de supervivencia de mayores de edad por clase:\n", survival_rate_adults_class)