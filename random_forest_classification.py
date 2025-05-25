# Clasificación con Random Forest

#Explicación:
#División de los datos: Al igual que en el modelo anterior, el dataset se divide en un conjunto de entrenamiento y uno de prueba.
#Escalado de características: Se realiza un escalado para garantizar que las características estén en la misma escala y evitar que un valor muy grande influencie más que otros.
#Entrenamiento del modelo Random Forest: Se utiliza el clasificador RandomForestClassifier, en este caso, con 10 estimadores (árboles) y el criterio de entropía para la división.
#Predicciones: Se realiza la predicción tanto para un nuevo ejemplo como para el conjunto de prueba.
#Evaluación del modelo: Se genera la matriz de confusión y se calcula la precisión para evaluar el rendimiento del modelo.
#Visualización: Se muestran las fronteras de decisión en el conjunto de entrenamiento y prueba, lo que ayuda a entender cómo el modelo clasifica los datos.

# Clasificación con Random Forest - Bank Marketing Dataset
# (Manteniendo estructura similar al código original)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from matplotlib.colors import ListedColormap

# Cargar dataset
dataset = pd.read_csv('bank.csv')

# Codificar variables categóricas (como en el original)
label_encoder = LabelEncoder()
dataset['deposit'] = label_encoder.fit_transform(dataset['deposit'])  # Target: 0=No, 1=Sí

# Seleccionar 2 características numéricas para mantener simplicidad visual (como en Social_Network_Ads)
X = dataset[['age', 'balance']].values  # Equivalente a 'Age' y 'EstimatedSalary' del original
y = dataset['deposit'].values

# Dividir datos (igual que el original)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Escalado (idéntico al original)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Configuración similar (solo cambiamos n_estimators a 100 para mejor performance)
classifier = RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=0)
classifier.fit(X_train, y_train)

# Predicción (ejemplo equivalente al original: age=30, balance=2000)
resultado = classifier.predict(sc.transform([[30, 2000]]))
print(f"Predicción para edad=30 y balance=2000: {'Sí' if resultado[0]==1 else 'No'}")

# Evaluación (idéntico al original)
y_pred = classifier.predict(X_test)
print("\nMatriz de Confusión:")
print(confusion_matrix(y_test, y_pred))
print(f"Precisión: {accuracy_score(y_test, y_pred):.2f}")

# Configuración de gráfico (misma estructura que el original pero con colores personalizados)
plt.figure(figsize=(10,6))

# Conjunto de ENTRENAMIENTO (como en el original pero con colores sólidos)
X_set, y_set = sc.inverse_transform(X_train), y_train
X1, X2 = np.meshgrid(
    np.arange(start=X_set[:, 0].min()-10, stop=X_set[:, 0].max()+10, step=1),
    np.arange(start=X_set[:, 1].min()-1000, stop=X_set[:, 1].max()+1000, step=500)
)
plt.contourf(X1, X2, classifier.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
             alpha=0.5, cmap=ListedColormap(['#FF0000', '#00FF00']))  # Rojo/Verde sólido

# Puntos de entrenamiento
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                color=['red', 'green'][i], 
                edgecolor='black',
                label=['No Deposit', 'Deposit'][i],
                s=30)

plt.title('Random Forest - Conjunto de Entrenamiento', fontsize=14)
plt.xlabel('Edad', fontsize=12)
plt.ylabel('Balance', fontsize=12)
plt.legend()
plt.show()

# Conjunto de PRUEBA
plt.figure(figsize=(10,6))
X_set, y_set = sc.inverse_transform(X_test), y_test
X1, X2 = np.meshgrid(
    np.arange(start=X_set[:, 0].min()-10, stop=X_set[:, 0].max()+10, step=1),
    np.arange(start=X_set[:, 1].min()-1000, stop=X_set[:, 1].max()+1000, step=500)
)
plt.contourf(X1, X2, classifier.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
             alpha=0.5, cmap=ListedColormap(['#FF0000', '#00FF00']))

for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                color=['red', 'green'][i],
                edgecolor='black',
                label=['No Deposit', 'Deposit'][i],
                s=30)

plt.title('Random Forest - Conjunto de Prueba', fontsize=14)
plt.xlabel('Edad', fontsize=12)
plt.ylabel('Balance', fontsize=12)
plt.legend()
plt.show()