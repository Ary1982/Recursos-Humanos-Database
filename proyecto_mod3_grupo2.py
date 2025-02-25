# PROYECTO: Optimización de Talento 
# - Análisis de Retención de Empleados

# Nombre equipo: TALENTIS

# ### IMPORTACIÓN LIBRERÍAS:

# Importamos las librerías que necesitamos

# Tratamiento de datos
# -----------------------------------------------------------------------
import pandas as pd
import numpy as np

# Visualización
# ------------------------------------------------------------------------------
import matplotlib.pyplot as plt
import seaborn as sns

# Evaluar linealidad de las relaciones entre las variables
# ------------------------------------------------------------------------------
from scipy.stats import shapiro, kstest

# Configuración
# -----------------------------------------------------------------------
pd.set_option('display.max_columns', None) # para poder visualizar todas las columnas de los DataFrames

# Gestión de los warnings
# -----------------------------------------------------------------------
import warnings
warnings.filterwarnings("ignore")

# CARGA DEL DATAFRAME:

df = pd.read_csv("HR_RAW_DATA.csv")
df.head (20)

# FUNCIÓN PARA EXPLORAR DATAFRAME

def explorar_dataset(df): # Función para una visión general del dataset

    print("Tamaño del dataset:")# Tamaño del dataset (filas y columnas)
    print(f"Filas: {df.shape[0]}, Columnas: {df.shape[1]}") 
    print("\nPrimeras 5 filas:") # Visualización primeras 5 filas
    display(df.head()) 
    print("\nTipos de datos:") # Tipos de datos por columna
    display(df.dtypes)
    print("\nValores_nulos:") 
    display(df.isnull().sum())

# Llamamos a la función para ver los datos
explorar_dataset(df)


# VISUALIZANDO PORCENTAJES NULOS:

def organizar_nulos(df):
   
    resumen = []

    for columna in df.columns:
        # Calculamos en una variable nueva el porcentaje de nulos de cada columna
        nulos_pct = (df[columna].isnull().sum() / len(df)) * 100
        
        # Incluimos solo columnas con nulos, diferenciando si son numéricas o categóricas
        if nulos_pct > 0:
            tipo = "Numérica" if df[columna].dtype in ['int64', 'float64'] else "Categórica"

            # Agregamos estos datos al resumen
            resumen.append({
                "Columna": columna,
                "Tipo": tipo,
                "Porcentaje_Nulos": nulos_pct
            })

    # Convertimos la lista a DataFrame y ordenamos por porcentaje de nulos (orden descendente)
    resumen_df = pd.DataFrame(resumen).sort_values(by="Porcentaje_Nulos", ascending=False)
    return resumen_df


resumen_nulos = organizar_nulos(df)
print(resumen_nulos)         


# ELIMINANDO COLUMNAS CON ALTO PORCENTAJE DE NULOS:

# Creamos una lista de columnas a eliminar
columnas_a_eliminar = ["NUMBERCHILDREN", "YearsInCurrentRole", "Department", "RoleDepartament", "StandardHours", "Over18"]

# Eliminamos las columnas directamente
df = df.drop(columns=columnas_a_eliminar)

# Mostramos el nuevo tamaño del DataFrame
print(f"Nuevo DataFrame con {df.shape[1]} columnas (después de eliminar {len(columnas_a_eliminar)} columnas).")

# Guardamos el DataFrame limpio en un nuevo archivo CSV
df.to_csv("HR_RAW_DATA_LIMPIO.csv", index=False)

print("Archivo guardado como 'HR_RAW_DATA_LIMPIO.csv'")

# NORMALIZANDO NOMBRES COLUMNAS

# Convertimos los nombres de las columnas a minúsculas
df.columns = df.columns.str.lower()

# Convertimos a minúsculas todas las columnas de tipo object 
for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].str.lower()

# Mostramos las primeras 20 filas sin la primera columna
df.head().iloc[:, 1:]

# Transformar los valores negativos a positivos usando abs()
df['distancefromhome'] = df['distancefromhome'].abs()

df.head().iloc[:, 1:]

# COMPROBANDO VALORES UNICOS EN MARITALSTATUS

df["maritalstatus"].unique()

# Como hay errores, los corregimos usando el metodo replace:

df['maritalstatus'].str.replace('marreid', 'married') #reemplaza una cadena o patrón en una columna

# Unificamos valores en la columna 'maritalstatus'
df["maritalstatus"] = df["maritalstatus"].replace({
    "marreid": "married",
    "divorced": "divorced",
})

# Verificamos valores únicos después de la corrección
print(df["maritalstatus"].unique())

# REEMPLAZAMOS LOS NUMEROS CON NOMBRES EN GENDER:

# Reemplazamos valores en la columna 'gender' (0 -> hombre, 1 -> mujer)
df["gender"] = df["gender"].replace({0: "hombre", 1: "mujer"})

# Verificamos los valores únicos después de la transformación
print(df["gender"].unique())

df["gender"].unique()

# COMPROBAMOS VALORES UNICOS EN LA COLUMNA AGE:

df["age"].unique()

# Unificamos valores en la columna 'age'
df["age"] = df["age"].replace({
    "forty-seven": "47",
    "fifty-eight": "58",
    "thirty-six": "36",
    "fifty-five": "55",
    "fifty-two": "52",
    "thirty-one": "31",
    "thirty": "30",
    "twenty-six": "26",
    "thirty-seven": "37",
    "thirty-two": "32",
    "twenty-four": "24",
})

# Verificamos valores únicos después de la corrección
print(df["age"].unique())

# VISUALIZAMOS VALORES UNICOS EN REMOTEWORK:

print(df["remotework"].unique())

# Normalizarmos la columna "remotework"

df["remotework"] = df["remotework"].replace({
    "1": "yes", "0": "no",
    "true": "yes", "false": "no"
})

# Verificamos los valores únicos después de la normalización
print(df["remotework"].unique())

df.head().iloc[:, 1:]

# TRANSFORMACION A INT64 DE VARIAS COLUMNAS:

def transformar_datos(df, columnas_a_cambiar):


    # Cambiamos el tipo de datos
    for col, tipo in columnas_a_cambiar.items():
        if tipo == "int64":  
            # Eliminamos comas antes de convertir a enteros
            df[col] = df[col].astype(str).str.replace(",0", "").astype(float).astype("Int64")
        else:
            df[col] = df[col].astype(tipo)

    # Estandarizamos nombres de columnas
    df.columns = [col.lower().replace(" ", "_") for col in df.columns]

    return df

# Llamamos a la función con corrección de formato
df = transformar_datos(df, columnas_a_cambiar={'employeenumber': 'int64'})
df = transformar_datos(df, columnas_a_cambiar={'worklifebalance': 'int64'})
df['totalworkingyears'] = df['totalworkingyears'].fillna(0)
df = transformar_datos(df, columnas_a_cambiar={'totalworkingyears': 'int64'})
df = transformar_datos(df, columnas_a_cambiar={'performancerating': 'int64'})
df['hourlyrate'] = df['hourlyrate'].replace('not available', pd.NA)
df['hourlyrate'] = df['hourlyrate'].fillna('0')
df = transformar_datos(df, columnas_a_cambiar={'hourlyrate': 'int64'})

# Calculamos la mediana
median_value = df['hourlyrate'].median()

# Reemplazamos los valores 0 por la mediana calculada
df['hourlyrate'] = df['hourlyrate'].replace(0, median_value)
df['sameasmonthlyincome'] = df['sameasmonthlyincome'].fillna('0')
df = transformar_datos(df, columnas_a_cambiar={'sameasmonthlyincome': 'int64'})
df = transformar_datos(df, columnas_a_cambiar={'age': 'int64'})



# Verificamos cambios
print(df.dtypes)
df.head(20)


print(df['hourlyrate'].unique())

# Eliminamos la columna "employeecount"
df = df.drop(columns=["employeecount"])

# Mostramos el DataFrame actualizado
print(df.head())  # Muestra las primeras 5 filas para verificar

df.head().iloc[:, 1:]

# Mostramos los valores más altos en la columna "employeenumber"
print(df["employeenumber"].max())

df_sorted = df.sort_values(by="employeenumber", ascending=False)

# Mostramos los valores únicos en la columna "employeenumber"
print(df["employeenumber"].unique())

# Obtenemos el valor máximo actual de "employeenumber"
max_value = df["employeenumber"].max()

# Generamos valores secuenciales para los nulos
contador = max_value + 1

# Reemplazamos los valores nulos con números secuenciales
df.loc[df["employeenumber"].isna(), "employeenumber"] = range(contador, contador + df["employeenumber"].isna().sum())

# Convertimos a tipo Int64 para evitar errores
df["employeenumber"] = df["employeenumber"].astype("Int64")

# Ordenamos el DataFrame por "employeenumber"
df_sorted = df.sort_values(by="employeenumber", ascending=True)

# Mostramos los valores ordenados
print(df_sorted[["employeenumber"]])

# Mostramos todos los valores únicos de la columna "employeenumber"
print(df_sorted["employeenumber"].unique())

print(df["attrition"].value_counts())

# AHORA LIMPIAMOS LAS COLUMNAS EDUCATION Y EDUCATIONFIELD:

# Mostramos las columnas "education" y "educationfield"
print(df[["education", "educationfield"]])

# Creamos un diccionario de mapeo para reemplazar los valores numéricos por etiquetas de educación
education_mapping = {
    1: "Secundaria",
    2: "FP",
    3: "Grado",
    4: "Master",
    5: "Doctorado"
}

# Aplicamos el mapeo a la columna "education"
df["education"] = df["education"].replace(education_mapping)

# Mostramos los resultados actualizados
print(df[["education"]].head(20))  # Muestra las primeras 20 filas para verificar

# Contamos la cantidad de valores para cada nivel de educación, incluyendo los nulos
education_counts = df["education"].value_counts(dropna=False)

# Reemplazamos NaN por "Nulos" en las etiquetas
labels = education_counts.index.astype(str)
labels = ["Nulos" if label == "nan" else label for label in labels]

# Creamos el gráfico de pastel
plt.figure(figsize=(8, 8))
plt.pie(education_counts, labels=labels, autopct='%1.1f%%', colors=plt.cm.Paired.colors, startangle=140)

# Título del gráfico
plt.title("Distribución del Nivel de Educación")

# Mostramos el gráfico
plt.show()


# Guardamos el DataFrame en el archivo original sobrescribiéndolo
df.to_csv("HR_RAW_DATA_LIMPIO.csv", index=False)

# Paso 1: Verificamos si 'educationfield' es nulo y creamos una columna booleana
df['educationfield_isnull'] = df['educationfield'].isnull()

# Paso 2: Agrupamos por 'education' y calculamos el porcentaje de nulos en 'educationfield'
porcentaje_nulos_por_categoria = df.groupby('education')['educationfield_isnull'].mean() * 100

# Mostramos el porcentaje de nulos por categoría de 'education'
print(porcentaje_nulos_por_categoria)

# Paso 3: Vemos las filas con 'educationfield' nulo
filas_nulas = df[df['educationfield'].isnull()]
print(filas_nulas[['education', 'educationfield']])

# eliminamos la columna booleana
df.drop(columns=['educationfield_isnull'], inplace=True)

# Reemplazamos valores NaN en la columna "educationfield" con "unknown"
df["educationfield"] = df["educationfield"].fillna("unknown")

# Mostramos los resultados actualizados
print(df[["education", "educationfield"]].head(20))  # Muestra las primeras 20 filas para verificar

# VOLVEMOS A COMPROBAR SI HAY NULOS EN EL NUEVO DF:

# Contamos los valores nulos en todo el DataFrame
nulos_totales = df.isnull().sum()

# Filtramos solo las columnas que tienen valores nulos (mayores que 0)
nulos_mayores_cero = nulos_totales[nulos_totales > 0]

# Mostramos los resultados
print(nulos_mayores_cero)

# Paso 1: Verificamos si 'businesstravel' es nulo y creamos una columna booleana
df['businesstravel_isnull'] = df['businesstravel'].isnull()

# Paso 2: Agrupamos por 'jobrole' y calculamos el porcentaje de nulos en 'businesstravel'
porcentaje_nulos_por_categoria = df.groupby('jobrole')['businesstravel_isnull'].mean() * 100

# Ordenamos de mayor a menor para visualizar mejor los resultados
porcentaje_nulos_por_categoria = porcentaje_nulos_por_categoria.sort_values(ascending=False)

# Mostramos el porcentaje de nulos por categoría de 'jobrole'
print(porcentaje_nulos_por_categoria)

# Paso 3: Vemos las filas con 'businesstravel' nulo
filas_nulas = df[df['businesstravel'].isnull()]
print(filas_nulas[['jobrole', 'businesstravel']])

# Eliminamos la columna booleana
df.drop(columns=['businesstravel_isnull'], inplace=True)

# Reemplazar valores nulos en "businesstravel" con "unknown"
df["businesstravel"] = df["businesstravel"].fillna("unknown")

# Verificamos los cambios
print(df["businesstravel"].value_counts())

df.head(10)

# VISUALIZAMOS LA DISTRIBUCIÓN DEL ESTADO CIVIL CON UN GRÁFICO PIE

# Contamos la cantidad de cada categoría en "maritalstatus"
marital_counts = df["maritalstatus"].value_counts()

# Creamos el gráfico de pastel
plt.figure(figsize=(8, 8))
plt.pie(marital_counts, labels=marital_counts.index, autopct='%1.1f%%', colors=plt.cm.Paired.colors, startangle=140)

# Título
plt.title("Distribución del Estado Civil (Marital Status)")

# Mostrar el gráfico
plt.show()

# ¿Cómo se distribuyen los clientes según su estado civil y género? 
# Vamos a visualizarlo con un gráfico de barras:

# Calculamos el conteo de clientes por estado civil y género
marital_gender_count = df.groupby(['maritalstatus', 'gender']).size().reset_index(name='Count')

# Ordenamos de mayor a menor por el número de clientes
marital_gender_count = marital_gender_count.sort_values(by='Count', ascending=False)

# Creamos el gráfico de barras con el DataFrame ordenado
sns.barplot(x='maritalstatus', y='Count', hue='gender', data=marital_gender_count, palette='Set2')

# Agregamos título y etiquetas
plt.title('Distribución de clientes por estado civil y género')
plt.xlabel('Estado Civil')
plt.ylabel('Número de Clientes')

# Activamos la cuadrícula
plt.grid(True, axis='y', linestyle='--', linewidth=0.5, alpha=0.7)

# Mostramos el gráfico
plt.show()


# ¿Cómo se distribuyen los clientes según su rol y género? 
# Vamos a verlo con un gráfico de barras:

# Calculamos el conteo de clientes por estado civil y género
marital_gender_count = df.groupby(['jobrole', 'gender']).size().reset_index(name='Count')

# Ordenar de mayor a menor por el número de clientes
marital_gender_count = marital_gender_count.sort_values(by='Count', ascending=False)

plt.figure(figsize=(12, 6))

# Crear el gráfico de barras con el DataFrame ordenado
sns.barplot(x='jobrole', y='Count', hue='gender', data=marital_gender_count, palette='Set2')

# Agregar título y etiquetas
plt.title('Distribución de clientes por estado civil y género')
plt.xlabel('Estado Civil')
plt.ylabel('Número de Clientes')

# Girar las etiquetas del eje X
plt.xticks(rotation=75) 

# Activar la cuadrícula
plt.grid(True, axis='y', linestyle='--', linewidth=0.5, alpha=0.7)

# Mostrar el gráfico
plt.show()

# Aquí nos dimos cuenta de que había duplicados que no tratamos:

# Duplicados

print(df[df.duplicated()]) #identifica duplicados

print('_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _', '\n')

print([df.duplicated().sum()]) #calcula la cantidad de filas duplicadas en el df

# Identificar duplicados en todas las columnas del DataFrame

duplicados_por_columna = df.copy()
for col in df.columns:
    duplicados_por_columna[col] = df[col].duplicated(keep=False)
# Filtrar solo las filas donde al menos una columna tiene un valor duplicado
df_duplicados = df[duplicados_por_columna.any(axis=1)]
# Mostrar los duplicados
print(df_duplicados)

# Eliminar la primera columna del DataFrame por posición (unkname_0)
df = df.drop(df.columns[0], axis=1)

# Mostrar las primeras filas para verificar
df.head()

df['salary'].describe().T

df.nunique()

df.dtypes

# Descubrimos que la columna Salary tiene solo un valor, así que no nos sirve:
# Podemos utilizar otras columnas para descubrir los sueldos de los empleados?
# - monthly income es lo que realmente percibe cada empleado
# - monthly rate es igual al salario fijo mensual base

# SEGUIMOS CON LA LIMPIEZA DE LAS COLUMNAS:

# Limpiar la columna 'dailyrate' para quitar el símbolo '$' y reemplazar la coma con punto
df['dailyrate'] = df['dailyrate'].replace({'\$': '', ',': '.'}, regex=True)

# Convertir la columna 'Dailyrate' de 'object' a 'float'
df['dailyrate'] = df['dailyrate'].astype(float)

# Multiplicar la columna 'Dailyrate' por 0.92
df['dailyrate'] = df['dailyrate'] * 0.92

# Mostrar el DataFrame resultante
print(df)

# CALCULAMOS SUELDOS:

#Calcular monthlysalary
df['monthlysalary'] = df['dailyrate'] * 22  # Basado en días hábiles

#Calcular annualsalary
df['annualsalary'] = df['dailyrate'] * 261  # Basado en días laborales del año

#Borrar columna salary
df.drop(columns=['salary'], inplace=True)

#Poner dailyrate con solo 2 decimales
df['dailyrate'] = df['dailyrate'].round(2)

# Mostrar el DataFrame resultante
df.head(10)

df['worklifebalance'].unique

# Rellenar los valores nulos con cero
df['worklifebalance'] = df['worklifebalance'].fillna(0)
df['worklifebalance'] = df['worklifebalance'].astype('int64')

# VAMOS A CONVERTIR A NUMERICAS LAS COLUMNAS DE LOS SUELDOS:

# Lista de columnas a convertir a float
columns_to_convert = ['distancefromhome', 'hourlyrate', 'monthlyincome', 'monthlyrate', 'worklifebalance']

# Verificar si las columnas existen antes de la conversión
columns_exist = [col for col in columns_to_convert if col in df.columns]

if not columns_exist:
    print("⚠️ Ninguna de las columnas especificadas existe en el DataFrame.")
else:
    # Convertir a float con manejo de errores
    df[columns_exist] = df[columns_exist].apply(pd.to_numeric, errors='coerce')

    # Verificar la conversión
    print(df[columns_exist].dtypes)


# GUARDAR EL DATAFRAME EN EL ARCHIVO ORIGINAL SOBRESCRIBIÉNDOLO
df.to_csv("HR_RAW_DATA_LIMPIO.csv", index=False)

df.head()

# PASOS A SEGUIR:

# Dividimos a los empleados en los grupos A y B según los criterios establecidos.
# Calculamos la tasa de rotación (porcentaje de empleados que dejaron la empresa) en cada grupo.
# Realizamos un análisis estadístico para determinar si hay una diferencia significativa en la tasa 
# de rotación entre los grupos A y B.
# Analizamos los resultados.
# Calculamos la magnitud de esta relación utilizando estadísticas como la diferencia de medias por ejemplo

# Clasificamos a los empleados en dos grupos según el valor de 'jobsatisfaction'

grupo_a = df[df['jobsatisfaction'] >= 3]  # Grupo A: Satisfacción >= 3
grupo_b = df[(df['jobsatisfaction'] < 3) & (df['jobsatisfaction'] > 0)]  # Grupo B: Satisfacción < 3 y > 0

# Calcular el porcentaje de 'Yes' en cada grupo
grupo_a_yes_percentage = (grupo_a['attrition'].value_counts(normalize=True).get('yes', 0)) * 100
grupo_b_yes_percentage = (grupo_b['attrition'].value_counts(normalize=True).get('yes', 0)) * 100

# Imprimir los resultados
print(f'Porcentaje de empleados con Attrition = Yes en Grupo A: {grupo_a_yes_percentage:.2f}%')
print(f'Porcentaje de empleados con Attrition = Yes en Grupo B: {grupo_b_yes_percentage:.2f}%')

# Contar la cantidad de empleados en cada grupo
grupo_a_count = len(grupo_a)
grupo_b_count = len(grupo_b)

# Crear un DataFrame para las proporciones
grupo_proporciones = pd.Series([grupo_a_count, grupo_b_count], index=['Grupo A (Control)', 'Grupo B (Variante)'])

# Gráfico de barras: Proporción de empleados en cada grupo
plt.figure(figsize=(8, 6))
sns.barplot(x=grupo_proporciones.index, y=grupo_proporciones.values, palette='Set2')

# Cambiar las etiquetas del eje Y a números del 1 al 100 (porcentajes)
plt.gca().set_yticklabels([f'{int(value / grupo_proporciones.sum() * 100)}%' for value in plt.gca().get_yticks()])

# Agregar título y etiquetas
plt.title('Proporción de Empleados por Nivel de Satisfacción')
plt.xlabel('Grupos')
plt.ylabel('Proporción de Empleados')

# Activar la cuadrícula
plt.grid(True, axis='y', linestyle='--', linewidth=0.5, alpha=0.7)

# Mostrar el gráfico
plt.show()

# Gráfico de pastel: Proporción de empleados en cada grupo
plt.figure(figsize=(8, 6))
colors = sns.color_palette('Set2', len(grupo_proporciones))

# Crear el gráfico de pastel
plt.pie(grupo_proporciones, labels=grupo_proporciones.index, autopct='%1.1f%%', colors=colors, startangle=90)

# Agregar título
plt.title('Proporción de Empleados por Nivel de Satisfacción')

# Mostrar el gráfico
plt.show()

# Clasificar a los empleados en dos grupos según el valor de 'jobsatisfaction'
grupo_a = df[df['jobsatisfaction'] >= 3]  # Grupo A: Satisfacción >= 3
grupo_b = df[(df['jobsatisfaction'] < 3) & (df['jobsatisfaction'] > 0)]  # Grupo B: Satisfacción < 3 y > 0

# Calcular el porcentaje de 'Yes' en cada grupo
grupo_a_yes_percentage = (grupo_a['attrition'].value_counts(normalize=True).get('yes', 0)) * 100
grupo_b_yes_percentage = (grupo_b['attrition'].value_counts(normalize=True).get('yes', 0)) * 100

# Imprimir los resultados
print(f'Porcentaje de empleados con Attrition = Yes en Grupo A: {grupo_a_yes_percentage:.2f}%')
print(f'Porcentaje de empleados con Attrition = Yes en Grupo B: {grupo_b_yes_percentage:.2f}%')

# Contar la cantidad de empleados en cada grupo
grupo_a_count = len(grupo_a)
grupo_b_count = len(grupo_b)

# Crear un DataFrame para las proporciones
grupo_proporciones = pd.Series([grupo_a_count, grupo_b_count], index=['Grupo A (Control)', 'Grupo B (Variante)'])

# Gráfico de barras: Proporción de empleados en cada grupo
plt.figure(figsize=(8, 6))
sns.barplot(x=grupo_proporciones.index, y=grupo_proporciones.values, palette='Set2')

# Cambiar las etiquetas del eje Y a números del 1 al 100 (porcentajes)
plt.gca().set_yticklabels([f'{int(value / grupo_proporciones.sum() * 100)}%' for value in plt.gca().get_yticks()])

# Agregar título y etiquetas
plt.title('Proporción de Empleados por Nivel de Satisfacción')
plt.xlabel('Grupos')
plt.ylabel('Proporción de Empleados')

# Activar la cuadrícula
plt.grid(True, axis='y', linestyle='--', linewidth=0.5, alpha=0.7)

# Mostrar el gráfico
plt.show()

# ---- NUEVO GRÁFICO: Cantidad de empleados por satisfacción y attrition ----

# Clasificar a los empleados en dos grupos según 'jobsatisfaction'
df['satisfaction_group'] = df['jobsatisfaction'].apply(lambda x: 'Grupo A (Control)' if x >= 3 else 'Grupo B (Variante)')

# Renombrar valores de attrition para que se muestren como 'Leave' y 'Stay'
df['Leave'] = df['attrition'].replace({'Yes': 'Leave', 'No': 'Stay'})

# Agrupar por los nuevos grupos y Leave
df_grouped = df.groupby(['satisfaction_group', 'Leave']).size().reset_index(name='Count')

# Calcular los porcentajes dentro de cada grupo
df_grouped['Percentage'] = df_grouped.groupby('satisfaction_group')['Count'].transform(lambda x: (x / x.sum()) * 100)

# Gráfico de barras apiladas en porcentaje
plt.figure(figsize=(8, 6))
sns.barplot(x='satisfaction_group', y='Percentage', hue='Leave', data=df_grouped, palette='Set2')

# Agregar título y etiquetas
plt.title('Distribución de Empleados por Grupos de Satisfacción y Leave (Porcentaje)')
plt.xlabel('Grupo de Satisfacción')
plt.ylabel('Porcentaje de Empleados')

# Mostrar los valores en cada barra
for container in plt.gca().containers:
    plt.bar_label(container, fmt='%.1f%%', label_type='edge', padding=3)

# Activar la cuadrícula
plt.grid(True, axis='y', linestyle='--', linewidth=0.5, alpha=0.7)

# Mostrar el gráfico
plt.show()



# GRAFICO PARA VISUALIZAR PORCENTAJE DE EMPLEADOS QUE DEJAN LA EMPRESA CON Y SIN TRABAJO REMOTO:

import seaborn as sns
import matplotlib.pyplot as plt

# Renombrar valores de attrition para que se muestren como 'Leave' y 'Stay'
df['Leave'] = df['attrition'].replace({'Yes': 'Leave', 'No': 'Stay'})

# Agrupar por 'remotework' y 'Leave'
df_grouped_remotework = df.groupby(['remotework', 'Leave']).size().reset_index(name='Count')

# Calcular los porcentajes dentro de cada grupo
df_grouped_remotework['Percentage'] = df_grouped_remotework.groupby('remotework')['Count'].transform(lambda x: (x / x.sum()) * 100)

# Gráfico de barras apiladas en porcentaje
plt.figure(figsize=(8, 6))
sns.barplot(x='remotework', y='Percentage', hue='Leave', data=df_grouped_remotework, palette='Set2')

# Agregar título y etiquetas
plt.title('Porcentaje de Empleados con o sin Trabajo Remoto que dejan la empresa')
plt.xlabel('Trabajo Remoto')
plt.ylabel('Porcentaje de Empleados')

# Mostrar los valores en cada barra
for container in plt.gca().containers:
    plt.bar_label(container, fmt='%.1f%%', label_type='edge', padding=3)

# Activar la cuadrícula
plt.grid(True, axis='y', linestyle='--', linewidth=0.5, alpha=0.7)

# Mostrar el gráfico
plt.show()


# -- GRAFICO PARA VISUALIZAR NIVEL EDUCATIVO SEGÚN SALARIO:

# Calcular el salario promedio por nivel educativo sin sobrescribir df original
salary_avg = df.groupby('education', as_index=False)['monthlysalary'].mean()

# Ordenar de mayor a menor por salario
salary_avg = salary_avg.sort_values(by='monthlysalary', ascending=False)

# Crear un gráfico de barras para mostrar el salario promedio por nivel educativo
plt.figure(figsize=(8, 6))
sns.barplot(x='education', y='monthlysalary', data=salary_avg, palette='plasma')

# Agregar título y etiquetas
plt.title('Salario Promedio por Nivel Educativo')
plt.xlabel('Nivel Educativo')
plt.ylabel('Salario Promedio')

# Girar las etiquetas del eje X para mejor visibilidad
plt.xticks(rotation=18)

# Mostrar el gráfico
plt.show()


# Gráfico para ver la distribución de empleados por años en la empresa y abandono del trabajo:

# Clasificar a los empleados en grupos según 'totalworkingyears'
df['years_group'] = df['totalworkingyears'].apply(lambda x: 'Grupo A (Control)' if x >= 3 else 'Grupo B (Variante)')

# Renombrar valores de attrition para que se muestren como 'Leave' y 'Stay'
df['Leave'] = df['attrition'].replace({'Yes': 'Leave', 'No': 'Stay'})

# Agrupar por los nuevos grupos de años en la empresa y Leave
df_grouped = df.groupby(['years_group', 'Leave']).size().reset_index(name='Count')

# Calcular los porcentajes dentro de cada grupo
df_grouped['Percentage'] = df_grouped.groupby('years_group')['Count'].transform(lambda x: (x / x.sum()) * 100)

# Gráfico de barras apiladas en porcentaje
plt.figure(figsize=(8, 6))
sns.barplot(x='years_group', y='Percentage', hue='Leave', data=df_grouped, palette='Set2')

# Agregar título y etiquetas
plt.title('Distribución de Empleados por Años en la Empresa y Leave (Porcentaje)')
plt.xlabel('Grupo de Años en la Empresa')
plt.ylabel('Porcentaje de Empleados')

# Mostrar los valores en cada barra
for container in plt.gca().containers:
    plt.bar_label(container, fmt='%.1f%%', label_type='edge', padding=3)

# Activar la cuadrícula
plt.grid(True, axis='y', linestyle='--', linewidth=0.5, alpha=0.7)

# Mostrar el gráfico
plt.show()


# Porcentaje de empleados que se van de la empresa según los años de antiguedad:

import matplotlib.pyplot as plt

# Calcular el porcentaje de 'Yes' en 'attrition' para cada 'totalworkingyears'
attrition_percentage = df.groupby('totalworkingyears')['attrition'].value_counts(normalize=True).unstack().reset_index()
attrition_percentage['Yes_Percentage'] = attrition_percentage['yes'] * 100  # Convertir a porcentaje

# Filtrar los datos para excluir el valor 40 y el valor 0 de 'totalworkingyears'
attrition_percentage = attrition_percentage[(attrition_percentage['totalworkingyears'] != 0) & 
                                            (attrition_percentage['totalworkingyears'] != 40)]

# Crear un mapa de color (colormap) que cambie dependiendo del valor de Y (Yes_Percentage)
norm = mpl.colors.Normalize(vmin=attrition_percentage['Yes_Percentage'].min(), vmax=attrition_percentage['Yes_Percentage'].max())
cmap = mpl.cm.viridis  # Se puede cambiar a cualquier otro mapa de colores

# Obtener la lista de colores basados en el colormap
colors = [cmap(norm(value)) for value in attrition_percentage['Yes_Percentage']]

# Crear el gráfico de barras
plt.figure(figsize=(10, 6))
sns.barplot(x='totalworkingyears', y='Yes_Percentage', data=attrition_percentage, palette=colors)

# Agregar etiquetas y título
plt.title('Porcentaje de Empleados que dejan la empresa, por Años en la Empresa')
plt.xlabel('Años en la Empresa')
plt.ylabel('Porcentaje de Empleados que se van')

# Girar las etiquetas del eje X
plt.xticks(rotation=90)

# Mostrar el gráfico
plt.show()



# Porcentaje de empleados que dejan la empresa según la distancia del puesto de trabajo:

# Calcular el porcentaje de 'Yes' en 'attrition' para cada 'distancefromhome'
attrition_percentage = df.groupby('distancefromhome')['attrition'].value_counts(normalize=True).unstack().reset_index()
attrition_percentage['Yes_Percentage'] = attrition_percentage['yes'] * 100  # Convertir a porcentaje

# Filtrar los datos para excluir el valor 40 y el valor 0 de 'distancefromhome'
attrition_percentage = attrition_percentage[(attrition_percentage['distancefromhome'] != 0) & 
                                            (attrition_percentage['distancefromhome'] != 40)]

# Crear un mapa de color (colormap) que cambie dependiendo del valor de Y (Yes_Percentage)
norm = mpl.colors.Normalize(vmin=attrition_percentage['Yes_Percentage'].min(), vmax=attrition_percentage['Yes_Percentage'].max())
cmap = mpl.cm.viridis  # Se puede cambiar a cualquier otro mapa de colores

# Crear el gráfico de barras
plt.figure(figsize=(10, 6))
sns.barplot(x='distancefromhome', y='Yes_Percentage', data=attrition_percentage, palette=cmap(norm(attrition_percentage['Yes_Percentage'])))

# Agregar etiquetas y título
plt.title('Porcentaje de Empleados que dejan la empresa, dependiendo de la distancia')
plt.xlabel('Distancia con el puesto de Trabajo')
plt.ylabel('Porcentaje de Empleados que se van')

# Girar las etiquetas del eje X
plt.xticks(rotation=90)

# Mostrar el gráfico
plt.show()


# Porcentaje de trabajadores que dejan la empresa según edad:

# Calcular el porcentaje de 'Yes' en 'attrition' para cada 'age'
attrition_percentage = df.groupby('age')['attrition'].value_counts(normalize=True).unstack().reset_index()
attrition_percentage['Yes_Percentage'] = attrition_percentage['yes'] * 100  # Convertir a porcentaje

# Filtrar los datos para excluir el valor 40 y el valor 0 de 'age'
attrition_percentage = attrition_percentage[(attrition_percentage['age'] != 0) & 
                                            (attrition_percentage['age'] != 40)]

# Crear un mapa de color (colormap) que cambie dependiendo del valor de Y (Yes_Percentage)
norm = mpl.colors.Normalize(vmin=attrition_percentage['Yes_Percentage'].min(), vmax=attrition_percentage['Yes_Percentage'].max())
cmap = mpl.cm.viridis  # Se puede cambiar a cualquier otro mapa de colores

# Crear el gráfico de barras
plt.figure(figsize=(10, 6))
sns.barplot(x='age', y='Yes_Percentage', data=attrition_percentage, palette=cmap(norm(attrition_percentage['Yes_Percentage'])))

# Agregar etiquetas y título
plt.title('Porcentaje de Empleados que dejan la empresa, dependiendo de la edad')
plt.xlabel('Edad del trabajador')
plt.ylabel('Porcentaje de empleados que se van')

# Girar las etiquetas del eje X
plt.xticks(rotation=90)

# Mostrar el gráfico
plt.show()



# Salario medio según Edad:

# Calcular la media de 'annualsalary' dependiendo de 'age'
salary_avg_by_age = df.groupby('age')['annualsalary'].mean().reset_index()

# Crear el gráfico de barras
plt.figure(figsize=(10, 6))
sns.barplot(x='age', y='annualsalary', data=salary_avg_by_age, palette='viridis')

# Agregar etiquetas y título
plt.title('Salario Anual Promedio Dependiendo de la Edad')
plt.xlabel('Edad del Trabajador')
plt.ylabel('Salario Anual Promedio')

# Girar las etiquetas del eje X si es necesario
plt.xticks(rotation=90)

# Mostrar el gráfico
plt.show()



# Porcentaje de empleados que dejan la empresa según los años pasados desde última promoción:

# Calcular el porcentaje de 'Yes' en 'attrition' para cada 'yearssincelastpromotion'
attrition_percentage = df.groupby('yearssincelastpromotion')['attrition'].value_counts(normalize=True).unstack().reset_index()
attrition_percentage['Yes_Percentage'] = attrition_percentage['yes'] * 100  # Convertir a porcentaje

# Filtrar los datos para excluir el valor 0 y el valor 40 de 'yearssincelastpromotion'
attrition_percentage = attrition_percentage[(attrition_percentage['yearssincelastpromotion'] != 0) & 
                                            (attrition_percentage['yearssincelastpromotion'] != 40)]

# Crear un mapa de color (colormap) que cambie dependiendo del valor de Y (Yes_Percentage)
norm = mpl.colors.Normalize(vmin=attrition_percentage['Yes_Percentage'].min(), vmax=attrition_percentage['Yes_Percentage'].max())
cmap = mpl.cm.viridis  # Se puede cambiar a cualquier otro mapa de colores

# Crear el gráfico de barras
plt.figure(figsize=(10, 6))
sns.barplot(x='yearssincelastpromotion', y='Yes_Percentage', data=attrition_percentage, palette=cmap(norm(attrition_percentage['Yes_Percentage'])))

# Agregar etiquetas y título
plt.title('Porcentaje de Empleados que dejan la empresa, dependiendo de los Años desde la Última Promoción')
plt.xlabel('Años desde la Última Promoción')
plt.ylabel('Porcentaje de Empleados que se van')

# Girar las etiquetas del eje X
plt.xticks(rotation=90)

# Mostrar el gráfico
plt.show()



# Detectados problemas al crear base de datos, hay duplicados en la columna employeenumber.
# Necesitamos eliminar los duplicados para poder usar la columna como ID (clave)

# Obtener los valores únicos que están duplicados
valores_duplicados_unicos = df['employeenumber'][df['employeenumber'].duplicated()].count()

# Mostrar los valores únicos duplicados
print(valores_duplicados_unicos)

# Cargar el archivo CSV
df = pd.read_csv('HR_RAW_DATA_LIMPIO.csv')

# Verificar si hay duplicados en la columna 'employeenumber'
duplicates = df[df.duplicated(subset='employeenumber', keep=False)]

# Mostrar los duplicados
if not duplicates.empty:
    print(f"Se encontraron {len(duplicates)} filas con 'employeenumber' duplicados:")
    print(duplicates[['employeenumber']])
else:
    print("No se encontraron duplicados en la columna 'employeenumber'.")


# Cargar el DataFrame
df = pd.read_csv('HR_RAW_DATA_LIMPIO.csv')

# Encontrar las filas donde 'employeenumber' está duplicado
duplicated_employeenumbers = df[df.duplicated(subset='employeenumber', keep=False)]

# Mostrar las filas duplicadas en 'employeenumber'
if not duplicated_employeenumbers.empty:
    print(f"Se encontraron {len(duplicated_employeenumbers)} filas con duplicados en 'employeenumber':")
    print(duplicated_employeenumbers[['employeenumber']])
else:
    print("No se encontraron duplicados en 'employeenumber'.")


# Encontrar las filas duplicadas completas para los 'employeenumber' duplicados
duplicated_complete_rows = duplicated_employeenumbers[duplicated_employeenumbers.duplicated(keep=False)]

# Mostrar las filas completas duplicadas
if not duplicated_complete_rows.empty:
    print(f"Se encontraron {len(duplicated_complete_rows)} filas completas duplicadas para los 'employeenumber' duplicados:")
    print(duplicated_complete_rows)
else:
    print("No se encontraron duplicados completos entre las filas con 'employeenumber' duplicados.")


# Cargar el DataFrame (ajusta la ruta si es necesario)
df = pd.read_csv('HR_RAW_DATA_LIMPIO.csv')

# Asegurarnos de que el número de filas en el DataFrame sea igual o menor que 1614
num_rows = len(df)

if num_rows > 1614:
    print("Error: El DataFrame tiene más de 1614 filas, no se puede asignar un número único para cada fila.")
else:
    # Crear un rango de números únicos de 1 a 1614
    unique_numbers = list(range(1, num_rows + 1))

    # Reemplazar los valores de 'employeenumber' con estos números únicos
    df['employeenumber'] = unique_numbers

    # Verificar que los valores son únicos
    if df['employeenumber'].is_unique:
        print("La columna 'employeenumber' ahora tiene valores únicos.")
    else:
        print("Hubo un problema al asignar valores únicos.")
    
    # Mostrar el DataFrame con los nuevos valores de 'employeenumber'
    print(df[['employeenumber']].head())


# Obtener los valores únicos que están duplicados
valores_duplicados_unicos = df['employeenumber'][df['employeenumber'].duplicated()].count()

# Mostrar los valores únicos duplicados
print(valores_duplicados_unicos)

#TRANSFORMAR CSV A ARCHIVO SQL

# Cargar el archivo CSV
csv_file = 'HR_RAW_DATA_LIMPIO.csv'
df = pd.read_csv(csv_file)

# Nombre de la tabla en SQL
table_name = 'empleados'

# Función para determinar el tipo de dato SQL adecuado
def get_sql_type(dtype):
    if pd.api.types.is_integer_dtype(dtype):
        return "INT"
    elif pd.api.types.is_float_dtype(dtype):
        return "FLOAT"
    elif pd.api.types.is_object_dtype(dtype):
        return "VARCHAR(255)"
    elif pd.api.types.is_datetime64_any_dtype(dtype):
        return "DATETIME"
    else:
        return "VARCHAR(255)"  # Por defecto

# Crear la sentencia SQL para la creación de la tabla
columns_sql = []
for column, dtype in df.dtypes.items():
    sql_type = get_sql_type(dtype)
    columns_sql.append(f"{column} {sql_type}")

# Crear la sentencia CREATE TABLE
create_table_sql = f"CREATE TABLE {table_name} (\n"
create_table_sql += ",\n".join(columns_sql)
create_table_sql += "\n);\n\n"

# Crear las sentencias INSERT INTO para cada fila de datos
insert_statements = []
for index, row in df.iterrows():
    values = ', '.join([f"'{str(value)}'" if isinstance(value, str) else str(value) for value in row])
    insert_statements.append(f"INSERT INTO {table_name} ({', '.join(df.columns)}) VALUES ({values});")

# Escribir todo en un archivo SQL
with open('HR_RAW_DATA_LIMPIO.sql', 'w', encoding='utf-8') as sql_file:
    sql_file.write(create_table_sql)
    sql_file.write("\n".join(insert_statements))

print("Archivo SQL generado exitosamente como HR_RAW_DATA_LIMPIO.sql")

# Leer el archivo SQL generado
with open('HR_RAW_DATA_LIMPIO.sql', 'r', encoding='utf-8') as sql_file:
    sql_content = sql_file.readlines()

# Filtrar las sentencias INSERT INTO
insert_statements = [line.strip() for line in sql_content if line.startswith('INSERT INTO')]

# Extraer los employeenumber de las sentencias INSERT INTO
employeenumbers = []
for statement in insert_statements:
    # Extraer los valores de los paréntesis de cada sentencia INSERT
    values = statement.split("VALUES")[1].strip()[1:-2]  # Elimina los paréntesis y comillas
    employeenumbers.append(int(values.split(",")[0]))  # El primer valor corresponde a 'employeenumber'

# Verificar duplicados en employeenumber
duplicates = [num for num in employeenumbers if employeenumbers.count(num) > 1]
if duplicates:
    print(f"Se encontraron duplicados: {set(duplicates)}")
else:
    print("No se encontraron duplicados.")


# Verificar si los valores de 'employeenumber' tienen espacios o caracteres extraños
df['employeenumber'] = df['employeenumber'].astype(str).str.strip()

# Verificar el tipo de dato de 'employeenumber'
print(df['employeenumber'].dtype)

# Cargar el archivo CSV
csv_file = 'HR_RAW_DATA_LIMPIO.csv'
df = pd.read_csv(csv_file)

# Eliminar filas duplicadas basadas en todas las columnas (asegurarse de que no haya duplicados generales)
df = df.drop_duplicates()

# Asegurarse de que la columna 'employeenumber' es única, si no, asignar valores únicos
if df['employeenumber'].duplicated().any():
    print("Se han encontrado duplicados en 'employeenumber'. Se corregirán.")
    # Crear nuevos valores únicos para 'employeenumber', si fuera necesario
    df['employeenumber'] = range(1, len(df) + 1)

# Nombre de la tabla en SQL
table_name = 'empleados'

# Función para determinar el tipo de dato SQL adecuado
def get_sql_type(dtype):
    if pd.api.types.is_integer_dtype(dtype):
        return "INT"
    elif pd.api.types.is_float_dtype(dtype):
        return "FLOAT"
    elif pd.api.types.is_object_dtype(dtype):
        return "VARCHAR(255)"
    elif pd.api.types.is_datetime64_any_dtype(dtype):
        return "DATETIME"
    else:
        return "VARCHAR(255)"  # Por defecto

# Crear la sentencia SQL para la creación de la tabla
columns_sql = []
for column, dtype in df.dtypes.items():
    sql_type = get_sql_type(dtype)
    columns_sql.append(f"{column} {sql_type}")

# Crear la sentencia CREATE TABLE
create_table_sql = f"CREATE TABLE {table_name} (\n"
create_table_sql += ",\n".join(columns_sql)
create_table_sql += "\n);\n\n"

# Crear las sentencias INSERT INTO para cada fila de datos
insert_statements = []
for index, row in df.iterrows():
    values = ', '.join([f"'{str(value)}'" if isinstance(value, str) else str(value) for value in row])
    insert_statements.append(f"INSERT INTO {table_name} ({', '.join(df.columns)}) VALUES ({values});")

# Escribir todo en un archivo SQL
with open('HR_RAW_DATA_LIMPIO.sql', 'w', encoding='utf-8') as sql_file:
    sql_file.write(create_table_sql)
    sql_file.write("\n".join(insert_statements))

print("Archivo SQL generado exitosamente como HR_RAW_DATA_LIMPIO.sql")



