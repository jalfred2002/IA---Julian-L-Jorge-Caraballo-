import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib

# Función de activación Sigmoide
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivada de la función Sigmoide
def sigmoid_derivative(x):
    return x * (1 - x)

# Función de activación ReLU
def relu(x):
    return np.maximum(0, x)

# Derivada de la función ReLU
def relu_derivative(x):
    return np.where(x > 0, 1, 0)

# Función de activación Tanh
def tanh(x):
    return np.tanh(x)

# Derivada de la función Tanh
def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2

# Clase MLP
class MLP:
    def __init__(self, input_size, hidden_sizes, output_size, activation='sigmoid'):
        """
        Inicializa el MLP con una o más capas ocultas.
        :param input_size: Número de características de entrada.
        :param hidden_sizes: Lista con el número de neuronas en cada capa oculta.
        :param output_size: Número de neuronas en la capa de salida.
        :param activation: Función de activación ('sigmoid', 'relu', 'tanh').
        """
        self.hidden_sizes = hidden_sizes
        self.activation = activation

        # Inicialización de pesos y sesgos
        self.weights = []
        self.biases = []

        # Capa de entrada a la primera capa oculta
        self.weights.append(np.random.randn(input_size, hidden_sizes[0]) * np.sqrt(2.0 / input_size))  # Inicialización He
        self.biases.append(np.zeros((1, hidden_sizes[0])))  # Sesgos inicializados en 0 (paréntesis corregidos)

        # Capas ocultas adicionales
        for i in range(1, len(hidden_sizes)):
            self.weights.append(np.random.randn(hidden_sizes[i-1], hidden_sizes[i]) * np.sqrt(2.0 / hidden_sizes[i-1]))
            self.biases.append(np.zeros((1, hidden_sizes[i])))

        # Última capa oculta a la capa de salida
        self.weights.append(np.random.randn(hidden_sizes[-1], output_size) * np.sqrt(2.0 / hidden_sizes[-1]))
        self.biases.append(np.zeros((1, output_size)))

    def forward(self, X):
        """
        Propagación hacia adelante.
        :param X: Datos de entrada.
        :return: Salida de la red.
        """
        self.layers = [X]  # Almacena las salidas de cada capa
        for i in range(len(self.weights)):
            layer_input = self.layers[-1] @ self.weights[i] + self.biases[i]  # Uso de @ para multiplicación de matrices
            layer_output = self._apply_activation(layer_input)
            self.layers.append(layer_output)
        return self.layers[-1]

    def _apply_activation(self, x):
        """
        Aplica la función de activación.
        """
        if self.activation == 'sigmoid':
            return sigmoid(x)
        elif self.activation == 'relu':
            return relu(x)
        elif self.activation == 'tanh':
            return tanh(x)
        else:
            raise ValueError("Función de activación no soportada")

    def backward(self, X, y, output):
        """
        Propagación hacia atrás y actualización de pesos.
        :param X: Datos de entrada.
        :param y: Etiquetas reales.
        :param output: Salida de la red.
        """
        # Cálculo del error en la capa de salida
        self.errors = [y - output]
        self.deltas = [self.errors[-1] * self._apply_activation_derivative(output)]

        # Propagación del error hacia atrás
        for i in range(len(self.weights) - 1, 0, -1):
            error = self.deltas[-1] @ self.weights[i].T  # Uso de @ para multiplicación de matrices
            delta = error * self._apply_activation_derivative(self.layers[i])
            self.errors.append(error)
            self.deltas.append(delta)

        # Invertir las listas para facilitar la actualización de pesos
        self.errors.reverse()
        self.deltas.reverse()

        # Actualización de pesos y sesgos
        for i in range(len(self.weights)):
            self.weights[i] += self.layers[i].T @ self.deltas[i]  # Uso de @ para multiplicación de matrices
            self.biases[i] += np.sum(self.deltas[i], axis=0, keepdims=True)

    def _apply_activation_derivative(self, x):
        """
        Aplica la derivada de la función de activación.
        """
        if self.activation == 'sigmoid':
            return sigmoid_derivative(x)
        elif self.activation == 'relu':
            return relu_derivative(x)
        elif self.activation == 'tanh':
            return tanh_derivative(x)
        else:
            raise ValueError("Función de activación no soportada")

    def train(self, X, y, epochs, learning_rate):
        """
        Entrena el MLP.
        :param X: Datos de entrenamiento.
        :param y: Etiquetas de entrenamiento.
        :param epochs: Número de épocas.
        :param learning_rate: Tasa de aprendizaje.
        """
        self.loss_history = []
        for epoch in range(epochs):
            output = self.forward(X)
            self.backward(X, y, output)
            loss = -np.mean(y * np.log(output + 1e-10) + (1 - y) * np.log(1 - output + 1e-10))  # Pérdida de entropía cruzada binaria
            self.loss_history.append(loss)
            if epoch % 100 == 0:
                print(f"Época {epoch}, Pérdida: {loss}")

    def predict(self, X):
        """
        Realiza predicciones.
        :param X: Datos de entrada.
        :return: Predicciones (0 o 1).
        """
        return (self.forward(X) > 0.5).astype(int)  # Usar un umbral de 0.5 para clasificación binaria

# Solicitar al usuario el nombre del archivo
nombre_archivo = input("Por favor, introduce el nombre del archivo CSV (con la extensión .csv): ")

# Validar que el archivo existe y es un CSV
if not os.path.isfile(nombre_archivo):
    print(f"Error: El archivo '{nombre_archivo}' no existe.")
    exit()
if not nombre_archivo.lower().endswith('.csv'):
    print(f"Error: El archivo '{nombre_archivo}' no es un archivo CSV.")
    exit()

try:
    # Cargar el dataset
    df = pd.read_csv(nombre_archivo)
    print(f"Archivo '{nombre_archivo}' cargado correctamente.")

    # Solicitar al usuario el nombre de la columna objetivo
    columna_objetivo = input("Por favor, introduce el nombre de la columna objetivo (la que indica si es fraude o no): ")

    # Verificar que la columna objetivo existe
    if columna_objetivo not in df.columns:
        print(f"Error: La columna '{columna_objetivo}' no existe en el archivo.")
        exit()

    # Verificar que la columna objetivo es binaria
    if not set(df[columna_objetivo]).issubset({0, 1}):
        print(f"Error: La columna '{columna_objetivo}' debe contener solo valores 0 y 1.")
        exit()

    # Separar características (X) y etiquetas (y)
    X = df.drop(columna_objetivo, axis=1)
    y = df[columna_objetivo].values

    # Verificar valores faltantes
    if df.isnull().any().any():
        print("Advertencia: El dataset contiene valores faltantes.")
        # Eliminar filas con valores faltantes
        df = df.dropna()
        if df.empty:
            print("Error: El dataset está vacío después de eliminar filas con valores faltantes.")
            exit()
        X = df.drop(columna_objetivo, axis=1)
        y = df[columna_objetivo].values

    # Preprocesamiento de datos
    # Normalización de características numéricas
    numeric_cols = X.select_dtypes(include=['float64', 'int64']).columns
    if len(numeric_cols) == 0:
        print("Advertencia: No hay columnas numéricas para normalizar.")
    else:
        scaler = StandardScaler()
        X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

    # Codificación de variables categóricas (si las hay)
    categorical_cols = X.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        encoder = OneHotEncoder(sparse=False, drop='first')  # Evita la multicolinealidad
        encoded_cols = encoder.fit_transform(X[categorical_cols])
        encoded_df = pd.DataFrame(encoded_cols, columns=encoder.get_feature_names_out(categorical_cols))
        X = X.drop(categorical_cols, axis=1)
        X = pd.concat([X, encoded_df], axis=1)
    else:
        print("Advertencia: No hay columnas categóricas para codificar.")

    # Dividir los datos en entrenamiento (60%), validación (20%) y prueba (20%)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

    if X_train.empty or X_val.empty or X_test.empty:
        print("Error: Uno de los conjuntos de datos (entrenamiento, validación o prueba) está vacío.")
        exit()

    # Hiperparámetros
    input_size = X_train.shape[1]  # Número de características
    hidden_sizes = [4, 4]  # Dos capas ocultas con 4 neuronas cada una
    output_size = 1  # 1 para clasificación binaria
    epochs = 1000  # Número de épocas de entrenamiento
    learning_rate = 0.01  # Tasa de aprendizaje

    # Inicializar el MLP con función de activación ReLU
    mlp = MLP(input_size, hidden_sizes, output_size, activation='relu')

    # Entrenar el MLP
    mlp.train(X_train, y_train, epochs, learning_rate)

    # Hacer predicciones en el conjunto de PRUEBA
    predictions = mlp.predict(X_test)

    # Convertir las etiquetas de prueba y las predicciones en tipo entero (0 o 1)
    y_test = y_test.astype(int)
    predictions = predictions.astype(int)

    # Crear un DataFrame con los resultados del conjunto de PRUEBA
    resultados = pd.DataFrame({'Real': y_test.flatten(), 'Predicción': predictions.flatten()})

    # Identificar las filas donde la predicción es fraude (1) y la realidad también (1)
    filas_fraude_detectado = resultados[(resultados['Real'] == 1) & (resultados['Predicción'] == 1)].index.tolist()

    # Si se detecta fraude, mostrar las filas correspondientes del DataFrame original
    if filas_fraude_detectado:
        print("\n¡FRAUDE DETECTADO!")
        print("Las siguientes filas del conjunto de PRUEBA son transacciones fraudulentas (según la IA):")
        for fila in filas_fraude_detectado:
            if fila < len(X_test):
                print(f"\nFila (índice del conjunto de PRUEBA): {fila}")
                print(X_test.iloc[fila])  # Usar .iloc para acceder a las filas correctamente
            else:
                print(f"\n¡ADVERTENCIA! El índice {fila} está fuera del rango del conjunto de PRUEBA.")
    else:
        print("\nNo se detectaron transacciones fraudulentas en el conjunto de PRUEBA.")

    # Evaluar el modelo
    if len(np.unique(y_test)) < 2:
        print("Advertencia: El conjunto de prueba no contiene ambas clases (0 y 1). No se pueden calcular todas las métricas.")
    else:
        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions)
        recall = recall_score(y_test, predictions)
        f1 = f1_score(y_test, predictions)

        print("\nMétricas del conjunto de PRUEBA:")
        print("Precisión:", accuracy)
        print("Recall:", recall)
        print("Puntuación F1:", f1)

    # Visualización de resultados
    # Matriz de confusión
    cm = confusion_matrix(y_test, predictions.astype(int))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Matriz de Confusión")
    plt.xlabel("Predicción")
    plt.ylabel("Real")
    plt.savefig('confusion_matrix.png')  # Guardar la matriz de confusión
    plt.show()

    # Curva de aprendizaje
    plt.plot(mlp.loss_history)
    plt.title("Curva de Aprendizaje")
    plt.xlabel("Época")
    plt.ylabel("Pérdida")
    plt.savefig('learning_curve.png')  # Guardar la curva de aprendizaje
    plt.show()

    # Guardar el modelo y los preprocesadores
    if os.path.exists('scaler.pkl') or os.path.exists('encoder.pkl') or os.path.exists('mlp_model.pkl'):
        overwrite = input("Los archivos ya existen. ¿Desea sobrescribirlos? (s/n): ")
        if overwrite.lower() != 's':
            print("Guardado cancelado.")
            exit()

    joblib.dump(scaler, 'scaler.pkl')
    if len(categorical_cols) > 0:
        joblib.dump(encoder, 'encoder.pkl')
    joblib.dump(mlp, 'mlp_model.pkl')
    print("Modelo y preprocesadores guardados correctamente.")

except pd.errors.EmptyDataError:
    print(f"Error: El archivo '{nombre_archivo}' está vacío.")
except pd.errors.ParserError:
    print(f"Error: El archivo '{nombre_archivo}' tiene un formato inválido.")
except UnicodeDecodeError:
    print(f"Error: El archivo '{nombre_archivo}' tiene problemas de codificación.")
except Exception as e:
    print(f"Error inesperado: {str(e)}")
    print(f"Tipo de error: {type(e).__name__}")