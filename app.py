from flask import Flask, request, jsonify
import pandas as pd
from pymongo import MongoClient

# Configuración de la conexión a la base de datos MongoDB
client = MongoClient("mongodb+srv://Miguel12195:uCBdaWo23Vln0gk8@cluster0.pa0yg19.mongodb.net/")
db = client['inteligentes']
datasetCollection = db['dataset']
trainCollection = db['train']

app = Flask(__name__)

@app.route('/')
def index():
    return '¡Hola, mundo!'

if __name__ == '__main__':
    app.run(debug=True)




#El sistema deberá permitir cargar documentos de Excel.
#Nota: Los archivos cargados deben ser almacenados en Mongo DB (Se recomienda utilizar Mongo Atlas) en formato JSON
@app.route('/load', methods=['POST'])
def load():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'error': 'No se proporcionó ningún archivo'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'Nombre de archivo no válido'}), 400

        if file:
            try:
                # Leer el archivo Excel usando pandas
                df = pd.read_excel(file)

                # Convertir los datos a formato JSON y almacenarlos en MongoDB
                data_as_json = df.to_dict(orient='records')
                datasetCollection.insert_many(data_as_json)

                return jsonify({'message': 'Datos cargados exitosamente a MongoDB'})
            except Exception as e:
                return jsonify({'error': f'Error al cargar el archivo: {str(e)}'}), 500


        return jsonify({'message': 'Solicitud POST recibida correctamente', 'data': data}), 200
    else:
         # Si la solicitud no es POST, devuelve un error
        return jsonify({'error': 'Se espera una solicitud POST'}), 400











#Devuelve los datos que genera el comando “describe” de pandas, por cada una de las columnas del dataset
@app.route('/basic_statistics/<dataset_id>', methods=['GET'])
def basic_statistics(dataset_id):
    if dataset_id == 'dataset1':
        # Cargar el dataset desde un archivo CSV usando pandas
        df = pd.read_csv('dataset1.csv')  # Asegúrate de cambiar el nombre del archivo

        # Obtener las estadísticas descriptivas usando describe para cada columna
        statistics = df.describe()

        # Convertir las estadísticas a un formato JSON y retornarlas
        return statistics.to_json()
    else:
        return jsonify({'error': 'No se encontró el dataset'}), 404









#Identificar los tipos de datos de las columnas (Texto, Numérico)
@app.route('/columns-describe/<dataset_id>', methods=['GET'])
def columns_describe(dataset_id):
    if dataset_id == 'dataset1':
        df = pd.read_csv('dataset1.csv')

        columns_data_types = {}
        return jsonify(columns_data_types)
    else:
        return jsonify({'error': 'No se encontró el dataset'}), 404










#Aplicar técnicas para tratamiento de datos faltantes, según el “tipo”
#seleccionado en el endpoint. Los tipos son los siguientes:
#1. Eliminar registros que contienen datos faltantes
#2. Imputación por media para variables numéricas, y moda por
#variables categóricas/texto
#Nota: Luego de realizar la imputación se debe crear una copia del
#dataset, esto quiere decir debe quedar la versión original del dataset
#y el nuevo con los datos imputados
@app.route('/imputation/<dataset_id>/type/<number_type>', methods=['POST'])
def imputation(dataset_id, number_type):
    # Verificar si el método de solicitud es POST
    if request.method == 'POST':
        data = request.json
        return jsonify({'message': 'Solicitud POST recibida correctamente', 'data': data}), 200
    else:
        # Si la solicitud no es POST, devuelve un error
        return jsonify({'error': 'Se espera una solicitud POST'}), 400









#Por cada una de las columnas se crean y almacenan en el servidor
#gráficos de:
# Histogramas
# Diagramas de caja (solo variables numéricas)
# Análisis de distribución de probabilidad
#Nota: en la respuesta del endpint se debe retornar la ruta en las que
#quedaron las imágenes guardadas. Se recomienda por cada dataset
#crear una carpeta con el nombre del identificador que la describe
@app.route('/general-univariate-graphs/<dataset_id>', methods=['POST'])
def general_univariate_graphs(dataset_id):
    # Verificar si el método de solicitud es POST
    if request.method == 'POST':
        data = request.json
        return jsonify({'message': 'Solicitud POST recibida correctamente', 'data': data}), 200
    else:
        # Si la solicitud no es POST, devuelve un error
        return jsonify({'error': 'Se espera una solicitud POST'}), 400










#Se debe crear y almacenar diagramas de Diagramas de caja y gráfico de densidad por clase
@app.route('/univariate-graphs-class/<dataset_id>/', methods=['POST'])
def univariate_graphs_class(dataset_id):
    if request.method == 'POST':
        data = request.json
        return jsonify({'message': 'Solicitud POST recibida correctamente', 'data': data}), 200
    else:
        # Si la solicitud no es POST, devuelve un error
        return jsonify({'error': 'Se espera una solicitud POST'}), 400









#Se debe retornar el enlace del servidor en el cual está almacenado el
#gráfico “pair plot” correspondiente el cruce de cada una de las
#columnas
@app.route('/bivariate-graphs-class/<dataset_id>/', methods=['GET'])
def bivariate_graphs_class(dataset_id):
    if dataset_id == 'dataset1':
        df = pd.read_csv('dataset1.csv')

        columns_data_types = {}
        return jsonify(columns_data_types)
    else:
        return jsonify({'error': 'No se encontró el dataset'}), 404










#Se debe retornar el enlace del servidor en el cual está almacenado el
#gráfico de correlación de las columnas numéricas
@app.route('/multivariate-graphs-class/<dataset_id>/', methods=['GET'])
def multivariate_graphs_class(dataset_id):
    if dataset_id == 'dataset1':
        df = pd.read_csv('dataset1.csv')

        columns_data_types = {}
        return jsonify(columns_data_types)
    else:
        return jsonify({'error': 'No se encontró el dataset'}), 404














#Según el dataset proporcionado por parámetro, se debe aplicar pca,
#como respuesta se debe retornar los pesos de las componentes, a su
#vez se debe crear una nueva versión del dataset con los datos
#transformados. El identificador del nuevo dataset también debe ser retornado
@app.route('/pca/<dataset_id>/', methods=['POST'])
def pca_analysis(dataset_id):
    if request.method == 'POST':
        data = request.json
        return jsonify({'message': 'Solicitud POST recibida correctamente', 'data': data}), 200
    else:
        # Si la solicitud no es POST, devuelve un error
        return jsonify({'error': 'Se espera una solicitud POST'}), 400















#Este debe recibir dentro del body de la petición los siguientes
#campos:
# “algorithms”: Es una lista de solo números los cuales
#determinan los algoritmos que se van a entrenar para el
#dataset brindado, seguir el siguiente orden:
#1. Regresión Logística
#2. KNN
#3. Máquinas de soporte vectorial
#4. Naive Bayes
#5. Árboles de decisión
#6. Redes neuronales multicapa
# “option_train”: Puede tomar uno de los siguientes valores
#numéricos
#1. Hold out (quiere decir que se debe aplicar partición
#70%-30%)
#2. Cross Validation (quiere decir que se deben utilizar
#5 folds)
# “normalization”: Puede tomar un valor numérico que
#representa
#1. MinMax
#2. Standar Scaler
#Cada entrenamiento genera un identificador, esto para poder
#diferenciar un entrenamiento de otro, cada entrenamiento tiene
@app.route('/train/<dataset_id>/', methods=['POST'])
def train(dataset_id):
    if request.method == 'POST':
        data = request.json
        return jsonify({'message': 'Solicitud POST recibida correctamente', 'data': data}), 200
    else:
        # Si la solicitud no es POST, devuelve un error
        return jsonify({'error': 'Se espera una solicitud POST'}), 400








#Debe permitir listar las métricas de cada uno de los modelos
#entrenados tales como:
# Matriz de confusión
# Accuracy
# Precision
# Recall
# F1 score
@app.route('/results/<train_id>', methods=['GET'])
def get_train_results(train_id):
    if train_id in []:
        # Si el train_id existe en los resultados simulados, se devuelve el resultado correspondiente
        return jsonify([train_id])
    else:
        # Si el train_id no existe, se devuelve un mensaje de error
        return jsonify({'error': 'No se encontraron resultados para el ID de entrenamiento especificado'}), 404










#Este permitirá realizar la predicción con el mejor modelo del
#entrenamiento en cuestión (basado en la métrica F1 Score). Nota
#recibe los parámetros de prueba dentro del body
@app.route('/prediction/<train_id>', methods=['GET'])
def get_predictions(train_id):
    if train_id == 'train_id':

        columns_data_types = {}
        return jsonify(columns_data_types)
    else:
        return jsonify({'error': 'No se encontró el dataset'}), 404
