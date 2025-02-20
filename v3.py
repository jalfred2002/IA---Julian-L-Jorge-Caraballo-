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
    def __init__(self, input_size, hidden_size, output_size, activation='sigmoid'):
        # Inicialización aleatoria de los pesos
        self.weights1 = np.random.rand(input_size, hidden_size)
        self.weights2 = np.random.rand(hidden_size, output_size)
        self.bias1 = np.zeros((1, hidden_size))
        self.bias2 = np.zeros((1, output_size))
        self.activation = activation

    def forward(self, X):
        # Propagación hacia adelante
        self.layer1 = self._apply_activation(np.dot(X, self.weights1) + self.bias1)
        self.layer2 = self._apply_activation(np.dot(self.layer1, self.weights2) + self.bias2)
        return self.layer2

    def _apply_activation(self, x):
        if self.activation == 'sigmoid':
            return sigmoid(x)
        elif self.activation == 'relu':
            return relu(x)
        elif self.activation == 'tanh':
            return tanh(x)
        else:
            raise ValueError("Función de activación no soportada")

    def backward(self, X, y, output):
        # Cálculo de errores y gradientes
        self.error2 = y - output
        self.delta2 = self.error2 * self._apply_activation_derivative(output)

        self.error1 = self.delta2.dot(self.weights2.T)
        self.delta1 = self.error1 * self._apply_activation_derivative(self.layer1)

        # Actualización de pesos y sesgos
        self.weights2 += self.layer1.T.dot(self.delta2)
        self.weights1 += X.T.dot(self.delta1)

        self.bias2 += np.sum(self.delta2, axis=0, keepdims=True)
        self.bias1 += np.sum(self.delta1, axis=0, keepdims=True)

    def _apply_activation_derivative(self, x):
        if self.activation == 'sigmoid':
            return sigmoid_derivative(x)
        elif self.activation == 'relu':
            return relu_derivative(x)
        elif self.activation == 'tanh':
            return tanh_derivative(x)
        else:
            raise ValueError("Función de activación no soportada")

    def train(self, X, y, epochs, learning_rate):
        # Entrenamiento del MLP
        self.loss_history = []
        for i in range(epochs):
            output = self.forward(X)
            self.backward(X, y, output)
            loss = np.mean(np.square(y - output))
            self.loss_history.append(loss)

    def predict(self, X):
        # Predicción del MLP
        return np.round(self.forward(X))

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

    # Separar características (X) y etiquetas (y)
    X = df.drop(columna_objetivo, axis=1)
    y = df[columna_objetivo].values

    # Verificar valores faltantes
    if df.isnull().any().any():
        print("Advertencia: El dataset contiene valores faltantes.")
        # Eliminar filas con valores faltantes
        df = df.dropna()
        X = df.drop(columna_objetivo, axis=1)
        y = df[columna_objetivo].values

    # Preprocesamiento de datos
    # Normalización de características numéricas
    numeric_cols = X.select_dtypes(include=['float64', 'int64']).columns
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

    # Dividir los datos en entrenamiento (60%), validación (20%) y prueba (20%)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

    # Hiperparámetros
    input_size = X_train.shape[1]  # Número de características
    hidden_size = 4  # Número de neuronas en la capa oculta
    output_size = 1  # 1 para clasificación binaria
    epochs = 100  # Número de épocas de entrenamiento
    learning_rate = 0.1  # Tasa de aprendizaje

    # Inicializar el MLP con función de activación ReLU
    mlp = MLP(input_size, hidden_size, output_size, activation='relu')

    # Entrenar el MLP
    mlp.train(X_train, y_train.reshape(-1, 1), epochs, learning_rate)

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
                print(X_test[fila])
            else:
                print(f"\n¡ADVERTENCIA! El índice {fila} está fuera del rango del conjunto de PRUEBA.")
    else:
        print("\nNo se detectaron transacciones fraudulentas en el conjunto de PRUEBA.")

    # Evaluar el modelo
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)

    print("\nMétricas del conjunto de PRUEBA:")
    print("Precisión:", accuracy)
    print("Precisión:", precision)
    print("Exhaustividad:", recall)
    print("Puntuación F1:", f1)

    # Visualización de resultados
    # Matriz de confusión
    cm = confusion_matrix(y_test, predictions)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Matriz de Confusión")
    plt.xlabel("Predicción")
    plt.ylabel("Real")
    plt.show()

    # Curva de aprendizaje
    plt.plot(mlp.loss_history)
    plt.title("Curva de Aprendizaje")
    plt.xlabel("Época")
    plt.ylabel("Pérdida")
    plt.show()

    # Guardar el modelo y los preprocesadores
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
    print(f"Error inesperado: {e}")