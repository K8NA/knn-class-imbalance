import openml
import warnings

from mla.knn import KNNClassifier
from mla.metrics import accuracy

from wei import KNNClassifier as clas
from wei import KNNRegressor as res

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score, recall_score, precision_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

import pandas as pd
import numpy as np
import scipy.stats as ss
import scikit_posthocs as sp
import matplotlib.pyplot as plt

#Suprimir avisos futuros
warnings.simplefilter(action='ignore', category=FutureWarning)

#Carregar dataset
dataset = openml.datasets.get_dataset(1464, download_data=True)
X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)

#Dividir em sets de treino e de teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=222)

#Inicializar o classificador KNN
knn = KNNClassifier(k=5)

#Treino
knn.fit(X_train, y_train)

#Predição
y_pred = knn.predict(X_test)


def run_benchmark(models, model_names, benchmark="OpenML-CC18"):
    results = pd.DataFrame(columns=["dataset", "model", "score"])  # create dataframe for results
    benchmark_suite = openml.study.get_suite(benchmark)  # obtain the benchmark suite

    # datasets IDs
    # 40983
    # 40994
    # 1464
    # 1487
    # 1494
    # 1489
    # 1068
    # 1067
    # 1063
    # 1053
    # 1050
    # 1049
    subset_benchmark_suite = benchmark_suite.tasks[0:10]

    # for task_id in benchmark_suite.tasks:  # iterate over all tasks
    for task_id in subset_benchmark_suite:  # iterate over subset tasks
        task = openml.tasks.get_task(task_id)  # download the OpenML task
        features, targets = task.get_X_and_y()  # get the data
        for model in range(len(models)):  # iterate over all models
            score = np.mean(cross_val_score(models[model], features, targets, cv=10, scoring="roc_auc_ovr"))
            if model_names:
                results = pd.concat(
                    [results, pd.DataFrame([[task_id, model_names[model], score]], columns=results.columns)],
                    ignore_index=True)
            else:
                results = pd.concat(
                    [results, pd.DataFrame([[task_id, str(models[model]), score]], columns=results.columns)],
                    ignore_index=True)
    results.to_csv("results.csv", index=False)


KNN = make_pipeline(SimpleImputer(strategy='constant'), StandardScaler(), KNeighborsClassifier())
KNN2 = make_pipeline(SimpleImputer(strategy='constant'), StandardScaler(), clas())
models = [KNN, KNN2]
model_names = ["KNN", "KNN2"]
run_benchmark(models=models, model_names=model_names)

results = pd.read_csv("results.csv")
avg_rank = results.groupby('dataset').score.rank(pct=True).groupby(results.model).mean()


test_results = sp.posthoc_conover_friedman(
    results,
    melted=True,
    block_col='dataset',
    group_col='model',
    y_col='score',
)
sp.sign_plot(test_results)


#Calcular a accuracy
acc = accuracy(y_test, y_pred)
print("Accuracy: ", acc)

def plot_cv(results_cv,metric='Accuracy'):
    fig, ax = plt.subplots()
    ax.boxplot(results_cv)
    ax.set_xticklabels(results_cv.columns)
    ax.set_ylabel(metric)
    ax.set_title('Cross-validation results for KNN and KNN2 in a dataset')
    plt.show()


# Calcular o F1 score5
f1 = f1_score(y_test, y_pred, average='weighted')
print("F1 Score: ", f1)

# Calcular e imprimir a precisão (precision), recall e F1 score para cada classe
precision = precision_score(y_test, y_pred, average=None)
recall = recall_score(y_test, y_pred, average=None)
f1_per_class = 2 * (precision * recall) / (precision + recall)
print("Precision per class:", precision)
print("Recall per class:", recall)
print("F1 Score per class:", f1_per_class)
