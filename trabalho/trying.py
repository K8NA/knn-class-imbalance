import openml
import sns
from matplotlib import pyplot as plt

from mla.knn import KNNClassifier
from mla.metrics import accuracy
from sklearn.model_selection import train_test_split
import warnings
import plotly.graph_objects as go
import numpy as np


#Suprimir avisos futuros (só porque me estava a incomodar, não é uma linha necessária)
warnings.simplefilter(action='ignore', category=FutureWarning)

#Carregar dataset
dataset = openml.datasets.get_dataset(11, download_data=True)
X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)

#Dividir em sets de treino e de teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#Inicializar o classificador KNN 
knn = KNNClassifier(k=5)

#Treino
knn.fit(X_train, y_train)

#Predição
y_pred = knn.predict(X_test)

#Calcular a accuracy
acc = accuracy(y_test, y_pred)
print("Accuracy:", acc)


