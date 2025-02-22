import numpy as np
import pandas as pd

# Función de activación Sigmoide
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivada de la función Sigmoide
def derivada_sigmoid(x):
    return x * (1 - x)

# Función de activación ReLU
def relu(x):
    return np.maximum(0, x)

# Derivada de la función ReLU
def derivada_relu(x):
    return np.where(x > 0, 1, 0)

# Clase MLP
class MLP:
    def __init__(self, tamaño_entrada, tamaños_ocultos, tamaño_salida, activacion='sigmoid'):
        """
        Inicializa el MLP con una o más capas ocultas.
        """
        self.tamaños_ocultos = tamaños_ocultos
        self.activacion = activacion

        # Inicialización de pesos y sesgos
        self.pesos = []
        self.sesgos = []

        # Capa de entrada a la primera capa oculta
        self.pesos.append(np.random.randn(tamaño_entrada, tamaños_ocultos[0]) * 0.01)  # Inicialización He
        self.sesgos.append(np.zeros((1, tamaños_ocultos[0])))

        # Capas ocultas adicionales
        for i in range(1, len(tamaños_ocultos)):
            self.pesos.append(np.random.randn(tamaños_ocultos[i-1], tamaños_ocultos[i]) * 0.01)
            self.sesgos.append(np.zeros((1, tamaños_ocultos[i])))

        # Última capa oculta a la capa de salida
        self.pesos.append(np.random.randn(tamaños_ocultos[-1], tamaño_salida) * 0.01)
        self.sesgos.append(np.zeros((1, tamaño_salida)))

    def propagacion_adelante(self, X):
        """
        Propagación hacia adelante.
        """
        self.capas = [X]  # Almacena las salidas de cada capa
        for i in range(len(self.pesos)):
            entrada_capa = self.capas[-1] @ self.pesos[i] + self.sesgos[i]  # Uso de @ para multiplicación de matrices
            salida_capa = self._aplicar_activacion(entrada_capa)
            self.capas.append(salida_capa)
        return self.capas[-1]

    def _aplicar_activacion(self, x):
        """
        Aplica la función de activación.
        """
        if self.activacion == 'sigmoid':
            return sigmoid(x)
        elif self.activacion == 'relu':
            return relu(x)
        else:
            raise ValueError("Función de activación no soportada")

    def propagacion_atras(self, X, y, salida, tasa_aprendizaje):
        """
        Propagación hacia atrás y actualización de pesos.
        """
        # Cálculo del error en la capa de salida
        self.errores = [y - salida]
        self.deltas = [self.errores[-1] * self._aplicar_derivada_activacion(salida)]

        # Propagación del error hacia atrás
        for i in range(len(self.pesos) - 1, 0, -1):
            error = self.deltas[-1] @ self.pesos[i].T  # Uso de @ para multiplicación de matrices
            delta = error * self._aplicar_derivada_activacion(self.capas[i])
            self.errores.append(error)
            self.deltas.append(delta)

        # Invertir las listas para facilitar la actualización de pesos
        self.errores.reverse()
        self.deltas.reverse()

        # Actualización de pesos y sesgos
        for i in range(len(self.pesos)):
            self.pesos[i] += tasa_aprendizaje * self.capas[i].T @ self.deltas[i]  # Uso de @ para multiplicación de matrices
            self.sesgos[i] += tasa_aprendizaje * np.sum(self.deltas[i], axis=0, keepdims=True)

    def _aplicar_derivada_activacion(self, x):
        """
        Aplica la derivada de la función de activación.
        """
        if self.activacion == 'sigmoid':
            return derivada_sigmoid(x)
        elif self.activacion == 'relu':
            return derivada_relu(x)
        else:
            raise ValueError("Función de activación no soportada")

    def entrenar(self, X, y, epocas, tasa_aprendizaje):
        """
        Entrena el MLP.
        """
        self.historial_perdida = []
        for epoca in range(epocas):
            salida = self.propagacion_adelante(X)
            self.propagacion_atras(X, y, salida, tasa_aprendizaje)
            perdida = -np.mean(y * np.log(salida + 1e-10) + (1 - y) * np.log(1 - salida + 1e-10))  # Pérdida de entropía cruzada binaria
            self.historial_perdida.append(perdida)
            if epoca % 100 == 0:
                print(f"Época {epoca}, Pérdida: {perdida}")

    def predecir(self, X):
        """
        Realiza predicciones.
        """
        return (self.propagacion_adelante(X) > 0.5).astype(int)  # Usar un umbral de 0.5 para clasificación binaria

def dividir_entrenamiento_prueba(X, y, tamaño_prueba=0.2, semilla_aleatoria=None):
    """
    Divide los datos en conjuntos de entrenamiento y prueba.
    Implementación básica sin sklearn.
    """
    if semilla_aleatoria is not None:
        np.random.seed(semilla_aleatoria)

    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)

    indice_division = int(X.shape[0] * (1 - tamaño_prueba))
    indices_entrenamiento, indices_prueba = indices[:indice_division], indices[indice_division:]

    X_entrenamiento, X_prueba = X[indices_entrenamiento], X[indices_prueba]
    y_entrenamiento, y_prueba = y[indices_entrenamiento], y[indices_prueba]

    return X_entrenamiento, X_prueba, y_entrenamiento, y_prueba


def escalar_estandar(X_entrenamiento, X_prueba):
    """
    Escala los datos para que tengan media 0 y desviación estándar 1.
    """
    media = np.mean(X_entrenamiento, axis=0)
    desviacion_estandar = np.std(X_entrenamiento, axis=0)
    X_entrenamiento_escalado = (X_entrenamiento - media) / (desviacion_estandar + 1e-8)  # Añade un pequeño valor para evitar división por cero
    X_prueba_escalado = (X_prueba - media) / (desviacion_estandar + 1e-8)
    return X_entrenamiento_escalado, X_prueba_escalado


def precision_exactitud(y_verdadero, y_predicho):
    """
    Calcula la precisión (accuracy).
    """
    predicciones_correctas = np.sum(y_verdadero == y_predicho)
    return predicciones_correctas / len(y_verdadero)

def precision_precision(y_verdadero, y_predicho):
    """
    Calcula la precisión (precision).
    """
    tp = np.sum((y_verdadero == 1) & (y_predicho == 1))
    fp = np.sum((y_verdadero == 0) & (y_predicho == 1))
    return tp / (tp + fp + 1e-8) # Sumar un pequeño valor para evitar la división por cero

def precision_exhaustividad(y_verdadero, y_predicho):
    """
    Calcula la exhaustividad (recall).
    """
    tp = np.sum((y_verdadero == 1) & (y_predicho == 1))
    fn = np.sum((y_verdadero == 1) & (y_predicho == 0))
    return tp / (tp + fn + 1e-8) # Sumar un pequeño valor para evitar la división por cero

def puntuacion_f1(y_verdadero, y_predicho):
    """
    Calcula la puntuación F1.
    """
    precision = precision_precision(y_verdadero, y_predicho)
    exhaustividad = precision_exhaustividad(y_verdadero, y_predicho)
    return 2 * (precision * exhaustividad) / (precision + exhaustividad + 1e-8) # Sumar un pequeño valor para evitar la división por cero

def matriz_confusion(y_verdadero, y_predicho):
    """
    Calcula la matriz de confusión.
    """
    etiquetas_unicas = np.unique(np.concatenate((y_verdadero, y_predicho)))
    matriz = np.zeros((len(etiquetas_unicas), len(etiquetas_unicas)), dtype=int)
    for i, etiqueta_verdadera in enumerate(etiquetas_unicas):
        for j, etiqueta_predicha in enumerate(etiquetas_unicas):
            matriz[i, j] = np.sum((y_verdadero == etiqueta_verdadera) & (y_predicho == etiqueta_predicha))
    return matriz

# Solicitar al usuario el nombre del archivo
nombre_archivo = input("Por favor, introduce el nombre del archivo CSV (con la extensión .csv): ")

try:
    # Cargar el dataset
    df = pd.read_csv(nombre_archivo)
    print(f"Archivo '{nombre_archivo}' cargado correctamente.")

    # Solicitar al usuario el nombre de la columna objetivo
    columna_objetivo = input("Por favor, introduce el nombre de la columna objetivo (la que indica si es fraude o no): ")

    # Separar características (X) y etiquetas (y)
    X = df.drop(columna_objetivo, axis=1).values  # Convertir a numpy array
    y = df[columna_objetivo].values

    # Dividir los datos en conjuntos de entrenamiento y prueba (80% entrenamiento, 20% prueba)
    X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = dividir_entrenamiento_prueba(X, y, tamaño_prueba=0.2, semilla_aleatoria=42)

    # Escalar los datos
    X_entrenamiento_escalado, X_prueba_escalado = escalar_estandar(X_entrenamiento, X_prueba)

    # Hiperparámetros
    tamaño_entrada = X_entrenamiento_escalado.shape[1]  # Número de características
    tamaños_ocultos = [4]  # Una capa oculta con 4 neuronas
    tamaño_salida = 1  # 1 para clasificación binaria
    epocas = 1000  # Número de épocas de entrenamiento
    tasa_aprendizaje = 0.001  # Tasa de aprendizaje

    # Inicializar el MLP
    mlp = MLP(tamaño_entrada, tamaños_ocultos, tamaño_salida, activacion='relu')

    # Entrenar el MLP
    mlp.entrenar(X_entrenamiento_escalado, y_entrenamiento.reshape(-1, 1), epocas, tasa_aprendizaje)

    # Hacer predicciones en el conjunto de PRUEBA
    predicciones = mlp.predecir(X_prueba_escalado)

    # Evaluar el modelo
    exactitud = precision_exactitud(y_prueba, predicciones.flatten())
    precision = precision_precision(y_prueba, predicciones.flatten())
    exhaustividad = precision_exhaustividad(y_prueba, predicciones.flatten())
    f1 = puntuacion_f1(y_prueba, predicciones.flatten())

    print("\nMétricas del conjunto de PRUEBA:")
    print("Precisión:", exactitud)
    print("Precisión:", precision)
    print("Exhaustividad:", exhaustividad)
    print("Puntuación F1:", f1)

    # Matriz de confusión
    matriz_de_confusion = matriz_confusion(y_prueba, predicciones.flatten())
    print("\nMatriz de Confusión:")
    print(matriz_de_confusion)

except FileNotFoundError:
    print(f"Error: El archivo '{nombre_archivo}' no se encontró.")
except Exception as e:
    print(f"Ocurrió un error: {e}")