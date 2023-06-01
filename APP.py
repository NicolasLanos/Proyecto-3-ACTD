import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import numpy as np
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator, BayesianEstimator
from pgmpy.inference import VariableElimination
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

# Cargamos los datos del archivo CSV
csv_path = "\\Users\\MAFISH\\Documents\\PROYECTOTRESACTD\\resultadosicfes.csv"

column_names = [
    "PERIODO", "ESTU_TIPODOCUMENTO", "ESTU_CONSECUTIVO", "COLE_AREA_UBICACION", "COLE_BILINGUE", "COLE_CALENDARIO",
    "COLE_CARACTER", "COLE_COD_DANE_ESTABLECIMIENTO", "COLE_COD_DANE_SEDE", "COLE_COD_DEPTO_UBICACION",
    "COLE_COD_MCPIO_UBICACION", "COLE_CODIGO_ICFES", "COLE_DEPTO_UBICACION", "COLE_GENERO", "COLE_JORNADA",
    "COLE_MCPIO_UBICACION", "COLE_NATURALEZA", "COLE_NOMBRE_ESTABLECIMIENTO", "COLE_NOMBRE_SEDE",
    "COLE_SEDE_PRINCIPAL", "ESTU_COD_DEPTO_PRESENTACION", "ESTU_COD_MCPIO_PRESENTACION", "ESTU_COD_RESIDE_DEPTO",
    "ESTU_COD_RESIDE_MCPIO", "ESTU_DEPTO_PRESENTACION", "ESTU_DEPTO_RESIDE", "ESTU_ESTADOINVESTIGACION",
    "ESTU_ESTUDIANTE", "ESTU_FECHANACIMIENTO", "ESTU_GENERO", "ESTU_MCPIO_PRESENTACION", "ESTU_MCPIO_RESIDE",
    "ESTU_NACIONALIDAD", "ESTU_PAIS_RESIDE", "ESTU_PRIVADO_LIBERTAD", "FAMI_CUARTOSHOGAR", "FAMI_EDUCACIONMADRE",
    "FAMI_EDUCACIONPADRE", "FAMI_ESTRATOVIVIENDA", "FAMI_PERSONASHOGAR", "FAMI_TIENEAUTOMOVIL", "FAMI_TIENECOMPUTADOR",
    "FAMI_TIENEINTERNET", "FAMI_TIENELAVADORA", "DESEMP_INGLES", "PUNT_INGLES", "PUNT_MATEMATICAS",
    "PUNT_SOCIALES_CIUDADANAS", "PUNT_C_NATURALES", "PUNT_LECTURA_CRITICA", "PUNT_GLOBAL"
]




df = pd.read_csv(csv_path)

#ELIMINAMOS COLUMNAS INNECESARIAS
columnas_eliminar = ['ESTU_TIPODOCUMENTO', 'ESTU_CONSECUTIVO', 'COLE_AREA_UBICACION','COLE_BILINGUE','COLE_CALENDARIO','COLE_CARACTER','COLE_COD_DANE_ESTABLECIMIENTO','COLE_COD_DANE_SEDE','COLE_COD_DEPTO_UBICACION','COLE_COD_MCPIO_UBICACION','COLE_CODIGO_ICFES','COLE_GENERO','COLE_JORNADA','COLE_MCPIO_UBICACION','COLE_NATURALEZA', 'COLE_NOMBRE_ESTABLECIMIENTO', 'COLE_NOMBRE_SEDE', 'COLE_SEDE_PRINCIPAL', 'ESTU_COD_DEPTO_PRESENTACION','ESTU_COD_MCPIO_PRESENTACION', 'ESTU_COD_RESIDE_DEPTO', 'ESTU_COD_RESIDE_MCPIO', 'ESTU_DEPTO_PRESENTACION', 'ESTU_DEPTO_RESIDE', 'ESTU_ESTADOINVESTIGACION', 'ESTU_ESTUDIANTE', 'ESTU_FECHANACIMIENTO','ESTU_MCPIO_PRESENTACION','ESTU_MCPIO_RESIDE', 'ESTU_NACIONALIDAD','ESTU_PAIS_RESIDE', 'ESTU_PRIVADO_LIBERTAD', 'DESEMP_INGLES','PUNT_INGLES','PUNT_MATEMATICAS','PUNT_SOCIALES_CIUDADANAS', 'PUNT_C_NATURALES', 'PUNT_LECTURA_CRITICA']

df = df.drop(columnas_eliminar, axis=1)


#PRIEMRA TRANSFORMACION DE DATOS

mapping = {'No': 0, 'Si': 1}

df['FAMI_TIENECOMPUTADOR'] = df['FAMI_TIENECOMPUTADOR'].map(mapping)

df['FAMI_TIENEINTERNET'] = df['FAMI_TIENEINTERNET'].map(mapping)

df['FAMI_TIENELAVADORA'] = df['FAMI_TIENELAVADORA'].map(mapping)

df['FAMI_TIENEAUTOMOVIL'] = df['FAMI_TIENEAUTOMOVIL'].map(mapping)

#SEGUNDA TRANSOFRMACION DE DATOS
mappping = {'Estrato 1': 1, 'Estrato 2': 2, 'Estrato 3': 3, 'Estrato 4': 4, 'Estrato 5': 5, 'Estrato 6': 6, 'Sin estrato': 0}

df['FAMI_ESTRATOVIVIENDA'] = df['FAMI_ESTRATOVIVIENDA'].map(mappping)


#TERCERA TRANSOFRMACION DE DATOS

mapppping = {'No sabe': 0,'No Aplica': 0,'Ninguno': 0, 'Primaria incompleta': 1, 'Primaria completa': 2, 'Secundaria (Bachillerato) incompleta': 3, 'Secundaria (Bachillerato) completa': 4, 'TÃ©cnica o tecnolÃ³gica incompleta': 5, 'TÃ©cnica o tecnolÃ³gica completa': 6, 'EducaciÃ³n profesional incompleta':7,'EducaciÃ³n profesional completa':8,'Postgrado':9}

df['FAMI_EDUCACIONMADRE'] = df['FAMI_EDUCACIONMADRE'].map(mapppping)
df['FAMI_EDUCACIONPADRE'] = df['FAMI_EDUCACIONPADRE'].map(mapppping)

#CUARTA TRANSFORMACION DE DATOS

mapeo = {'F': 0, 'M': 1}

df['ESTU_GENERO'] = df['ESTU_GENERO'].map(mapeo)

#QUINTA TRANSFORMACION DE DATOS

mappeo = {'Uno': 1, 'Dos': 2,'Tres':3, 'Cuatro':4,'Cinco':5, 'Seis o mas':6}

df['FAMI_CUARTOSHOGAR'] = df['FAMI_CUARTOSHOGAR'].map(mappeo)

#SEXTA TRANSFORMACION DE DATOS

mapppeo = {'1 a 2': 1, '3 a 4': 2,'5 a 6':3, '7 a 8':4,'9 o mÃ¡s':5}

df['FAMI_PERSONASHOGAR'] = df['FAMI_PERSONASHOGAR'].map(mapppeo)



# MATRIZ DE CORRELACION
corr_matrix = df.corr()
# GRAFICA MATRIZ DE CORRELACION


plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()

