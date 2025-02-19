import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Función de activación Sigmoide
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivada de la función Sigmoide
def sigmoid_derivative(x):
    return x * (1 - x)

# Clase MLP
class MLP:
    def __init__(self, input_size, hidden_size, output_size):
        # Inicialización aleatoria de los pesos
        self.weights1 = np.random.rand(input_size, hidden_size)
        self.weights2 = np.random.rand(hidden_size, output_size)
        self.bias1 = np.zeros((1, hidden_size))
        self.bias2 = np.zeros((1, output_size))

    def forward(self, X):
        # Propagación hacia adelante
        self.layer1 = sigmoid(np.dot(X, self.weights1) + self.bias1)
        self.layer2 = sigmoid(np.dot(self.layer1, self.weights2) + self.bias2)
        return self.layer2

    def backward(self, X, y, output):
        # Cálculo de errores y gradientes
        self.error2 = y - output
        self.delta2 = self.error2 * sigmoid_derivative(output)

        self.error1 = self.delta2.dot(self.weights2.T)
        self.delta1 = self.error1 * sigmoid_derivative(self.layer1)

        # Actualización de pesos y sesgos
        self.weights2 += self.layer1.T.dot(self.delta2)
        self.weights1 += X.T.dot(self.delta1)

        self.bias2 += np.sum(self.delta2, axis=0, keepdims=True)
        self.bias1 += np.sum(self.delta1, axis=0, keepdims=True)

    def train(self, X, y, epochs, learning_rate):
        # Entrenamiento del MLP
        for i in range(epochs):
            output = self.forward(X)
            self.backward(X, y, output)

    def predict(self, X):
        # Predicción del MLP
        return np.round(self.forward(X))

# Solicitar al usuario el nombre del archivo
nombre_archivo = input("Por favor, introduce el nombre del archivo CSV (con la extensión .csv): ")

try:
    # Cargar el dataset
    df = pd.read_csv(nombre_archivo)
    print(f"Archivo '{nombre_archivo}' cargado correctamente.")

    # Solicitar al usuario el nombre de la columna objetivo
    columna_objetivo = input("Por favor, introduce el nombre de la columna objetivo (la que indica si es fraude o no): ")

    # Separar características (X) y etiquetas (y)
    try:
        X = df.drop(columna_objetivo, axis=1).values
        y = df[columna_objetivo].values
        print(f"Columna objetivo '{columna_objetivo}' separada correctamente.")
    except KeyError:
        print(f"Error: La columna '{columna_objetivo}' no existe en el archivo.")
        exit()  # Salir del programa si la columna no existe

    # Dividir los datos en conjuntos de entrenamiento y prueba (80% entrenamiento, 20% prueba)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Hiperparámetros
    input_size = X_train.shape[1]  # Número de características
    hidden_size = 4  # Número de neuronas en la capa oculta
    output_size = 1  # 1 para clasificación binaria
    epochs = 100  # Número de épocas de entrenamiento
    learning_rate = 0.1  # Tasa de aprendizaje

    # Inicializar el MLP
    mlp = MLP(input_size, hidden_size, output_size)

    # Entrenar el MLP
    mlp.train(X_train, y_train.reshape(-1, 1), epochs, learning_rate)

    # Hacer predicciones en el conjunto de PRUEBA
    predictions = mlp.predict(X_test)

    #Convertir las etiquetas de prueba y las predicciones en tipo entero (0 o 1)
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
            # Asegurarse de que el índice no exceda las dimensiones de X_test
            if fila < len(X_test):
                # Mostrar la fila del conjunto de PRUEBA con todas las características
                print(f"\nFila (índice del conjunto de PRUEBA): {fila}")
                print(X_test[fila])  # Mostrar todas las características de esa fila
            else:
                print(f"\n¡ADVERTENCIA! El índice {fila} está fuera del rango del conjunto de PRUEBA.")
    else:
        print("\nNo se detectaron transacciones fraudulentas en el conjunto de PRUEBA.")

    # Evaluar el modelo
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)

    print("\nMetricas del conjunto de PRUEBA:")
    print("Precisión:", accuracy)
    print("Precisión:", precision)
    print("Exhaustividad:", recall)
    print("Puntuación F1:", f1)

except FileNotFoundError:
    print(f"Error: El archivo '{nombre_archivo}' no se encontró.")
except Exception as e:
    print(f"Ocurrió un error: {e}")