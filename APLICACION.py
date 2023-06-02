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
from sklearn.model_selection import train_test_split
from pgmpy.estimators import HillClimbSearch
from pgmpy.estimators import K2Score
import networkx as nx
import matplotlib.pyplot as plt
from dash_bootstrap_components import Tabs, Tab

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
df =df.dropna()

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

Nuevos_nombres = ['Periodo', 'Departamento', 'Genero','Cuartos','Madre','Padre','Estrato','Conviven','Carro','Computador','Internet','Lavadora','Puntaje']

df.columns = Nuevos_nombres

df.drop('Departamento', axis=1, inplace=True)

df.drop('Periodo', axis=1, inplace=True)

































#INICIAMOS CON LA CREACION DEL MODELO     SE DIVIDE EL SET EN DATOS DE TEST Y DE PRUEBA

train_og, test_og = train_test_split(df, test_size=0.3)

#SE CREAN AMBOS GRUPOS DE DATOS

train = train_og.dropna()
test = test_og.dropna()

#NOS ASEGURAMOS DE NO TENER COLUMNAS NO DESEADAS EN NUESTRO ESTUDIO


#CONVERTIMOS LA VARIABLE DE INTERES EN BINARIA PARA LOS DATOS DE PRUEBA Y LOS DE TEST


train.loc[train['Puntaje'] < 359, 'Puntaje'] = 0
train.loc[train['Puntaje'] >= 359, 'Puntaje'] = 1







#ESTABLECEMOS LOS METODOS DE PUNTAJE

scoring_method = K2Score(data=df)
esth = HillClimbSearch(data=df)

#CREAMOS EL NUEVO MODELO UTILIZANDO PUNTAKE K2

new_model = esth.estimate(
    scoring_method=scoring_method,
    max_indegree=1,
    max_iter=int(1e6),black_list=(
        tuple(['Puntaje', 'Departamento']),
        tuple(['Puntaje', 'Genero']),
        tuple(['Puntaje', 'Cuartos']),
        tuple(['Puntaje', 'Madre']),
        tuple(['Puntaje', 'Padre']),
        tuple(['Puntaje', 'Estrato']),
        tuple(['Puntaje', 'Conviven']),
        tuple(['Puntaje', 'Carro']),
        tuple(['Puntaje', 'Computador']),
        tuple(['Puntaje', 'Internet']),
        tuple(['Puntaje', 'Lavadora']),
        tuple(['Puntaje', 'Puntaje']),        

    ))

#IMPRIMIMOS MODELO PARA VERIFICAR
print(new_model)
print(new_model.nodes())
print(new_model.edges())



# Definición del DAG
nodes = ['Genero', 'Cuartos', 'Madre', 'Padre', 'Estrato', 'Conviven', 'Carro', 'Computador', 'Internet', 'Lavadora', 'Puntaje']
edges = [('Cuartos', 'Conviven'), ('Madre', 'Padre'), ('Madre', 'Puntaje'), ('Madre', 'Genero'), ('Padre', 'Estrato'), ('Estrato', 'Internet'), ('Computador', 'Carro'), ('Internet', 'Computador'), ('Internet', 'Lavadora'), ('Lavadora', 'Cuartos')]














###################################################################################################################

def bayesian_inference(Genero, Cuartos, Madre, Padre, Estrato, Conviven, Carro, Computador, Internet, Lavadora):
    modelHill = BayesianNetwork(edges)
    modelHill.fit(train, estimator=MaximumLikelihoodEstimator)
    infer = VariableElimination(modelHill)
    evidence = {'Genero': Genero, 'Cuartos': Cuartos, 'Madre': Madre, 'Padre': Padre, 'Estrato': Estrato, 'Conviven': Conviven, 'Carro': Carro, 'Computador': Computador,'Internet':Internet,'Lavadora':Lavadora}
    q = infer.query(['Puntaje'], evidence=evidence)
    return q

##############################################################################################################



















# Define the Bayesian inference function
def bayesian_inference(Genero, Cuartos, Madre, Padre, Estrato, Conviven, Carro, Computador, Internet, Lavadora):
    modelHill = BayesianNetwork(edges)
    modelHill.fit(train, estimator=MaximumLikelihoodEstimator)
    infer = VariableElimination(modelHill)
    evidence = {'Genero': Genero, 'Cuartos': Cuartos, 'Madre': Madre, 'Padre': Padre, 'Estrato': Estrato,
                'Conviven': Conviven, 'Carro': Carro, 'Computador': Computador, 'Internet': Internet,
                'Lavadora': Lavadora}
    q = infer.query(['Puntaje'], evidence=evidence)
    return q

# Create the Dash application
app = dash.Dash(__name__)

# Define the layout of the application
app.layout = html.Div([
    html.H1('Aplicación de Inferencia Bayesiana'),
    dcc.Tabs([
        dcc.Tab(label='Tab 1', children=[
            # Add your visualization components here
        ]),
        dcc.Tab(label='Tab 2', children=[
            html.Div([
                html.H1('Aplicación de Inferencia Bayesiana'),
                dcc.Input(id='genero-input', type='text', placeholder='Género'),
                dcc.Input(id='cuartos-input', type='number', placeholder='Cuartos'),
                dcc.Input(id='madre-input', type='text', placeholder='Madre'),
                dcc.Input(id='padre-input', type='text', placeholder='Padre'),
                dcc.Input(id='estrato-input', type='number', placeholder='Estrato'),
                dcc.Input(id='conviven-input', type='number', placeholder='Conviven'),
                dcc.Input(id='carro-input', type='number', placeholder='Carro'),
                dcc.Input(id='computador-input', type='number', placeholder='Computador'),
                dcc.Input(id='internet-input', type='number', placeholder='Internet'),
                dcc.Input(id='lavadora-input', type='number', placeholder='Lavadora'),
                html.Button('Calcular', id='calcular-button'),
                html.Div(id='output')
            ])
        ])
    ])
])


@app.callback(Output('output', 'children'),
              [Input('calcular-button', 'n_clicks')],
              [State('genero-input', 'value'),
               State('cuartos-input', 'value'),
               State('madre-input', 'value'),
               State('padre-input', 'value'),
               State('estrato-input', 'value'),
               State('conviven-input', 'value'),
               State('carro-input', 'value'),
               State('computador-input', 'value'),
               State('internet-input', 'value'),
               State('lavadora-input', 'value')])
def calculate_inference(n_clicks, genero, cuartos, madre, padre, estrato, conviven, carro, computador, internet, lavadora):
    result = bayesian_inference(genero, cuartos, madre, padre, estrato, conviven, carro, computador, internet, lavadora)
    return f"The probability of Puntaje = 1 given the evidence is: {result['Puntaje'][1]}"


if __name__ == '__main__':
    app.run_server(debug=True)