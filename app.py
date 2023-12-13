from flask import Flask, request, jsonify
import pandas as pd
from pymongo import MongoClient
from gridfs import GridFS
import seaborn as sns # pip install seaborn
import matplotlib.pyplot as plt
from io import BytesIO
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler # pip install scikit-learn
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
import joblib
import numpy as np
from bson import json_util


# Configuración de la conexión a la base de datos MongoDB
client = MongoClient("mongodb+srv://Miguel12195:uCBdaWo23Vln0gk8@cluster0.pa0yg19.mongodb.net/")
db = client['inteligentes']
datasets = client['datasets']
datasets_imputed = client['datasets_imputed']
datasets_pca= client['datasets_pca']
datasetCollection = db['dataset']
trainDb = client['train']
fs = GridFS(db)

app = Flask(__name__)

@app.route('/')
def index():
    return '¡Hola, mundo!'

if __name__ == '__main__':
    app.run(debug=True, ssl_context='adhoc')




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
                df = pd.read_csv(file)
                json_data = df.to_dict(orient="records")
                filename = file.filename.rsplit('.', 1)[0]
                collection = datasets[filename]
                collection.insert_many(json_data)
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
    collection = datasets[dataset_id]
    
    if collection.name not in datasets.list_collection_names():
        return jsonify({'error': 'No se encontró el dataset'}), 404
    
    documents = collection.find()
    df = pd.DataFrame(documents)
    statistics = df.describe()
    
    return statistics.to_json()









#Identificar los tipos de datos de las columnas (Texto, Numérico)
@app.route('/columns-describe/<dataset_id>', methods=['GET'])
def columns_describe(dataset_id):
    collection = datasets[dataset_id]
    
    if collection.name not in datasets.list_collection_names():
        return jsonify({'error': 'No se encontró el dataset'}), 404
    
    documents = collection.find()
    df = pd.DataFrame(documents)
    columns_data_types = df.dtypes.apply(lambda x: 'Numérico' if pd.api.types.is_numeric_dtype(x) else 'Texto')

    return jsonify(columns_data_types.to_dict())









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
    if request.method == 'POST':
        collection = datasets[dataset_id]
        
        if collection.name not in datasets.list_collection_names():
            return jsonify({'error': 'No se encontró el dataset'}), 404
        
        documents = collection.find()
        df = pd.DataFrame(documents)

        # Copia del DataFrame
        df_imputed = df.copy()

        if number_type == '1':
            df_imputed.dropna(inplace=True)
        elif number_type == '2':
            for column in df_imputed.columns:
                if pd.api.types.is_numeric_dtype(df_imputed[column]):
                    df_imputed[column].fillna(df_imputed[column].mean(), inplace=True)
                else:
                    df_imputed[column].fillna(df_imputed[column].mode()[0], inplace=True)

        coleccion_imputed = datasets_imputed[f'{collection.name}_imputed_{number_type}']
        coleccion_imputed.insert_many(df_imputed.to_dict('records'))

        return jsonify({'message': 'Imputación realizada correctamente', 'imputed_dataset': coleccion_imputed.name}), 200
    else:
        # Si la solicitud no es POST, devuelve un error
        return jsonify({'error': 'Se espera una solicitud POST'}), 400










#Por cada una de las columnas se crean y almacenan en el servidor
#gráficos de:
#- Histogramas
#- Diagramas de caja (solo variables numéricas)
#- Análisis de distribución de probabilidad
#Nota: en la respuesta del endpint se debe retornar la ruta en las que
#quedaron las imágenes guardadas. Se recomienda por cada dataset
#crear una carpeta con el nombre del identificador que la describe
@app.route('/general-univariate-graphs/<dataset_id>', methods=['POST'])
def general_univariate_graphs(dataset_id):
    if request.method == 'POST':
        collection = datasets_imputed[dataset_id]
        
        if collection.name not in datasets_imputed.list_collection_names():
            return jsonify({'error': 'No se encontró el dataset'}), 404
        
        documents = collection.find()
        df = pd.DataFrame(documents)

        # Directorio relativo para almacenar las imágenes
        relative_images_dir = os.path.join(os.getcwd(), 'images\\' + collection.name)
        os.makedirs(relative_images_dir, exist_ok=True)

        # Diccionario para almacenar las rutas de las imágenes
        image_paths = {}

        # Generar y guardar gráficos para cada columna
        for column in df.columns:
            if column == '_id':
                continue

            # Histograma
            plt.figure()
            sns.histplot(df[column], kde=False)
            plt.title(f'Histograma de {column}')
            graph_type_dir = os.path.join(relative_images_dir, 'histograms')
            os.makedirs(graph_type_dir, exist_ok=True)
            histogram_path = os.path.join(graph_type_dir, f'histogram_{column}.png')
            plt.savefig(histogram_path, format='png')

            # Almacenar la ruta en el diccionario
            image_paths[f'histogram_{column}'] = histogram_path

            ## Diagrama de caja (solo para variables numéricas)
            if pd.api.types.is_numeric_dtype(df[column]):
                plt.figure()
                sns.boxplot(x=df[column])
                plt.title(f'Diagrama de Caja de {column}')
                graph_type_dir = os.path.join(relative_images_dir, 'boxplots')
                os.makedirs(graph_type_dir, exist_ok=True)
                boxplot_path = os.path.join(graph_type_dir, f'boxplot_{column}.png')
                plt.savefig(boxplot_path, format='png')

                # Almacenar la ruta en el diccionario
                image_paths[f'boxplot_{column}'] = boxplot_path

            # Análisis de distribución de probabilidad (utilizando seaborn)
            plt.figure()
            sns.displot(df[column], kde=True)
            plt.title(f'Análisis de Distribución de Probabilidad de {column}')
            graph_type_dir = os.path.join(relative_images_dir, 'distplots')
            os.makedirs(graph_type_dir, exist_ok=True)
            distplot_path = os.path.join(graph_type_dir, f'distplot_{column}.png')
            plt.savefig(distplot_path, format='png')

            # Almacenar la ruta en el diccionario
            image_paths[f'distplot_{column}'] = distplot_path

        return jsonify({
            'message': 'Gráficos generados y guardados correctamente localmente',
            'dataset_id': dataset_id,
            'image_paths': image_paths
        }), 200
    else:
        # Si la solicitud no es POST, devuelve un error
        return jsonify({'error': 'Se espera una solicitud POST'}), 400










#Se debe crear y almacenar diagramas de Diagramas de caja y gráfico de densidad por clase
@app.route('/univariate-graphs-class/<dataset_id>/', methods=['POST'])
def univariate_graphs_class(dataset_id):
    if request.method == 'POST':
        collection = datasets_imputed[dataset_id]
        
        if collection.name not in datasets_imputed.list_collection_names():
            return jsonify({'error': 'No se encontró el dataset'}), 404
        
        documents = collection.find()
        df = pd.DataFrame(documents)

        # Directorio relativo para almacenar las imágenes
        relative_images_dir = os.path.join(os.getcwd(), 'images', collection.name)
        os.makedirs(relative_images_dir, exist_ok=True)

        # Diccionario para almacenar las rutas de las imágenes
        image_paths = {}

        # Obtener la última columna (clases)
        classes_column = df.columns[-1]

        # Obtener los valores únicos en la última columna
        unique_classes = df[classes_column].unique()

        # Generar y guardar gráficos para cada clase
        for class_value in unique_classes:
            # Filtrar el DataFrame por la clase actual
            class_df = df[df[classes_column] == class_value]

            # Directorio específico para la clase
            class_dir = os.path.join(relative_images_dir, str(class_value))
            os.makedirs(class_dir, exist_ok=True)

            # Generar y guardar boxplot para cada columna numérica
            for column in class_df.select_dtypes(include='number').columns:
                if column == '_id':
                    continue
                
                plt.figure()
                sns.boxplot(x=class_df[column])
                plt.title(f'Diagrama de Caja de {column} para la clase {class_value}')
                boxplot_path = os.path.join(class_dir, f'boxplot_{column}.png')
                plt.savefig(boxplot_path, format='png')

                # Almacenar la ruta en el diccionario
                image_paths[f'boxplot_{column}_class_{class_value}'] = boxplot_path

            # Generar y guardar diagrama de densidad para cada columna
            for column in class_df.select_dtypes(include='number').columns:
                if column == '_id':
                    continue

                plt.figure()
                sns.kdeplot(class_df[column], fill=True)
                plt.title(f'Diagrama de Densidad de {column} para la clase {class_value}')
                densityplot_path = os.path.join(class_dir, f'densityplot_{column}.png')
                plt.savefig(densityplot_path, format='png')

                # Almacenar la ruta en el diccionario
                image_paths[f'densityplot_{column}_class_{class_value}'] = densityplot_path

        return jsonify({
            'message': 'Gráficos generados y guardados correctamente localmente por clases',
            'dataset_id': dataset_id,
            'image_paths': image_paths
        }), 200
    else:
        # Si la solicitud no es POST, devuelve un error
        return jsonify({'error': 'Se espera una solicitud POST'}), 400









#Se debe retornar el enlace del servidor en el cual está almacenado el
#gráfico “pair plot” correspondiente el cruce de cada una de las
#columnas
@app.route('/bivariate-graphs-class/<dataset_id>/', methods=['GET'])
def bivariate_graphs_class(dataset_id):
    if request.method == 'GET':
        collection = datasets_imputed[dataset_id]

        if collection.name not in datasets_imputed.list_collection_names():
            return jsonify({'error': 'No se encontró el dataset'}), 404

        documents = collection.find()
        df = pd.DataFrame(documents)

        relative_images_dir = os.path.join(os.getcwd(), 'images', collection.name)
        os.makedirs(relative_images_dir, exist_ok=True)

        image_paths = {}

        plt.figure(figsize=(12, 8))
        sns.pairplot(df, hue=df.columns[-1])

        plt.title('Pair Plot para el cruce de cada columna')
        pairplot_path = os.path.join(relative_images_dir, 'pairplot.png')
        plt.savefig(pairplot_path, format='png')

        image_paths['pairplot'] = pairplot_path

        return jsonify({
            'message': 'Pair Plot generado y guardado correctamente localmente',
            'dataset_id': dataset_id,
            'image_paths': image_paths
        }), 200
    else:
        return jsonify({'error': 'Se espera una solicitud GET'}), 400










#Se debe retornar el enlace del servidor en el cual está almacenado el
#gráfico de correlación de las columnas numéricas
@app.route('/multivariate-graphs-class/<dataset_id>/', methods=['GET'])
def multivariate_graphs_class(dataset_id):
    if request.method == 'GET':
        collection = datasets_imputed[dataset_id]

        if collection.name not in datasets_imputed.list_collection_names():
            return jsonify({'error': 'No se encontró el dataset'}), 404

        documents = collection.find()
        df = pd.DataFrame(documents)

        # Directorio relativo para almacenar las imágenes
        relative_images_dir = os.path.join(os.getcwd(), 'images', collection.name)
        os.makedirs(relative_images_dir, exist_ok=True)

        # Diccionario para almacenar las rutas de las imágenes
        image_paths = {}

        # Filtrar las columnas numéricas para el gráfico de correlación
        numeric_columns = df.select_dtypes(include='number').columns

        # Generar el gráfico de correlación
        correlation_matrix = df[numeric_columns].corr()
        plt.figure(figsize=(12, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")

        # Configuraciones del gráfico
        plt.title('Gráfico de Correlación de Columnas Numéricas')
        correlation_plot_path = os.path.join(relative_images_dir, 'correlation_plot.png')
        plt.savefig(correlation_plot_path, format='png')

        # Almacenar la ruta en el diccionario
        image_paths['correlation_plot'] = correlation_plot_path

        return jsonify({
            'message': 'Gráfico de correlación generado y guardado correctamente localmente',
            'dataset_id': dataset_id,
            'image_paths': image_paths
        }), 200
    else:
        # Si la solicitud no es GET, devuelve un error
        return jsonify({'error': 'Se espera una solicitud GET'}), 400














#Según el dataset proporcionado por parámetro, se debe aplicar pca,
#como respuesta se debe retornar los pesos de las componentes, a su
#vez se debe crear una nueva versión del dataset con los datos
#transformados. El identificador del nuevo dataset también debe ser retornado
@app.route('/pca/<dataset_id>/', methods=['POST'])
def pca_analysis(dataset_id):
    if request.method == 'POST':
        collection = datasets_imputed[dataset_id]

        if collection.name not in datasets_imputed.list_collection_names():
            return jsonify({'error': 'No se encontró el dataset'}), 404

        documents = collection.find()
        df = pd.DataFrame(documents)

        if '_id' in df.columns:
            df = df.drop(columns=['_id'])
            
        X = df.drop(columns=[df.columns[-1]])
        y = df[df.columns[-1]]

        X_normalized = StandardScaler().fit_transform(X)

        pca = PCA()
        pca.fit(X_normalized)
        
        component_weights = pca.components_

        X_transformed = pca.transform(X_normalized)

        df_transformed = pd.DataFrame(X_transformed, columns=[f'{column}_pca' for column in X.columns])
        df_transformed[df.columns[-1]] = y 

        new_collection_name = f'{collection.name}_pca'
        new_collection = datasets_pca[new_collection_name]
        new_collection.insert_many(df_transformed.to_dict(orient='records'))

        return jsonify({
            'message': 'PCA aplicado con éxito',
            'component_weights': component_weights.tolist(),
            'new_dataset_id': new_collection_name
        }), 200
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
#asociado un dataset y la lista de modelos que se utilizaron
@app.route('/train/<dataset_id>/', methods=['POST'])
def train(dataset_id):
    if request.method == 'POST':
        algorithms_str = request.form.get('algorithms')
        algorithms = [int(number) for number in algorithms_str.split(", ")]
        option_train = int(request.form.get('option_train'))
        normalization = int(request.form.get('normalization'))

        # Cargar el dataset desde MongoDB
        collection = datasets_imputed[dataset_id]
        documents = collection.find()
        df = pd.DataFrame(documents)

        # Excluir la columna '_id'
        if '_id' in df.columns:
            df = df.drop(columns=['_id'])

        # Obtener características (X) y variable objetivo (y)
        X = df.drop(columns=[df.columns[-1]])
        y = df[df.columns[-1]]

        # Normalización
        if normalization == 1:
            scaler = MinMaxScaler()
            X_normalized = scaler.fit_transform(X)
        elif normalization == 2:
            scaler = StandardScaler()
            X_normalized = scaler.fit_transform(X)
        else:
            return jsonify({'error': 'Opción de normalización no válida'}), 400

        # Dividir datos en conjunto de entrenamiento y prueba
        if option_train == 1:
            X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.3, random_state=42)
        elif option_train == 2:
            X_train, X_test, y_train, y_test = X_normalized, None, y, None
        else:
            return jsonify({'error': 'Opción de entrenamiento no válida'}), 400

        # Inicializar modelos
        models = {
            1: LogisticRegression(),
            2: KNeighborsClassifier(),
            3: SVC(),
            4: GaussianNB(),
            5: DecisionTreeClassifier(),
            6: MLPClassifier()
        }
        
        # Inicializar métricas
        collection = datasets_imputed[dataset_id]
        
        metrics_collection = trainDb['metrics']
        best_collection = trainDb['best']
        best_model = {'algorithm': None, 'metrics': None}

        # Entrenar modelos seleccionados y guardar identificadores
        training_ids = []
        for algorithm in algorithms:
            if algorithm in models:
                model = models[algorithm]
                model.fit(X_train, y_train)
                
                # Calcular métricas
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='weighted')
                recall = recall_score(y_test, y_pred, average='weighted')
                f1 = f1_score(y_test, y_pred, average='weighted')
                confusion = confusion_matrix(y_test, y_pred).tolist()
        
                # Almacenar métricas en la colección 'metrics'
                metrics_doc = {
                    'train_id': f'model_{algorithm}_{dataset_id}',
                    'algorithm': algorithm,
                    'dataset_id': dataset_id,
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'confusion_matrix': confusion
                }
                metrics_collection.insert_one(metrics_doc)

                # Actualizar el mejor modelo si es necesario (basado en F1 score)
                if best_model['algorithm'] is None or f1 > best_model['metrics']['f1']:
                    best_model['algorithm'] = algorithm
                    best_model['metrics'] = {
                        'accuracy': accuracy,
                        'precision': precision,
                        'recall': recall,
                        'f1': f1,
                        'confusion_matrix': confusion
                    }
                    best_model['train_id'] = f'model_{algorithm}_{dataset_id}'

                # Guardar el modelo entrenado en una nueva colección
                with BytesIO() as model_bytes_io:
                    joblib.dump(model, model_bytes_io)
                    model_bytes = model_bytes_io.getvalue()

                # Guardar el modelo entrenado en GridFS
                fs.put(model_bytes, filename=f'model_{algorithm}_{dataset_id}')

                # Guardar el identificador del entrenamiento
                training_ids.append({'algorithm': algorithm, 'training_id': f'model_{algorithm}_{dataset_id}'})
            else:
                return jsonify({'error': f'Algoritmo {algorithm} no válido'}), 400

        best_collection.update_one(
            {'dataset_id': dataset_id},
            {'$set': {'best_model': best_model}},
            upsert=True
        )
        return jsonify({'message': 'Entrenamiento completado exitosamente', 'training_ids': training_ids}), 200
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
    if trainDb['metrics'].find_one({'train_id': train_id}):
        metrics_docs = trainDb['metrics'].find_one({'train_id': train_id})
        return jsonify(json_util.dumps(metrics_docs))
    else:
        return jsonify({'error': f'No se encontraron métricas para train_id: {train_id}'}), 404




#Este permitirá realizar la predicción con el mejor modelo del
#entrenamiento en cuestión (basado en la métrica F1 Score). Nota
#recibe los parámetros de prueba dentro del body
@app.route('/prediction/<train_id>', methods=['POST'])
def get_predictions(train_id):
    if request.method == 'POST':
        # Obtener los datos de prueba del cuerpo de la solicitud
        test_data = request.get_json()

        test_values = [float(value) for value in test_data.values()]
        # Buscar el mejor modelo asociado al train_id
        best_model_entry = trainDb['best'].find_one({'best_model.train_id': train_id})

        if best_model_entry:
            # Cargar el mejor modelo desde GridFS
            best_model_filename = best_model_entry['best_model']['train_id']
            best_model_bytes = fs.get_last_version(best_model_filename).read()
            best_model = joblib.load(BytesIO(best_model_bytes))

            # Realizar predicción con el mejor modelo
            prediction = best_model.predict([test_values])[0]

            return jsonify({'prediction': prediction})
        else:
            return jsonify({'error': 'No se encontró el mejor modelo para el train_id especificado'}), 404
    else:
        return jsonify({'error': 'Se espera una solicitud POST'}), 400
