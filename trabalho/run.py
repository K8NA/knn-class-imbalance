import openml
from mla.knn import KNNClassifier
from mla.metrics import accuracy
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, recall_score, precision_score
import warnings

#Suprimir avisos futuros (só porque me estava a incomodar, não é uma linha necessária)
warnings.simplefilter(action='ignore', category=FutureWarning)

#Carregar dataset
dataset = openml.datasets.get_dataset(1464, download_data=True)
X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)

#Dividir em sets de treino e de teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=222)


#Inicializar o classificador KNN 
knn = KNNClassifier(k=5)

#Treino
knn.fit(X_train, y_train)

#Predição
y_pred = knn.predict(X_test)

#Calcular a accuracy
acc = accuracy(y_test, y_pred)
print("Accuracy:", acc)

# Calcular o F1 score5
f1 = f1_score(y_test, y_pred, average='weighted')
print("F1 Score:", f1)

# Calcular e imprimir a precisão (precision), recall e F1 score para cada classe
precision = precision_score(y_test, y_pred, average=None)
recall = recall_score(y_test, y_pred, average=None)
f1_per_class = 2 * (precision * recall) / (precision + recall)
print("Precision per class:", precision)
print("Recall per class:", recall)
print("F1 Score per class:", f1_per_class)
