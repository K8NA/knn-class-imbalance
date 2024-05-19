import openml
import warnings

from mla.knn import KNNClassifier as mlaKNNClassifier
from mla.metrics import accuracy

from wei import KNNClassifier as weiKNNClassifier
from wei import KNNRegressor

from sklearn.model_selection import train_test_split, cross_val_score
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

warnings.simplefilter(action='ignore', category=FutureWarning)

# Load dataset
dataset = openml.datasets.get_dataset(1464, download_data=True)
X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=222)

# Initialize and train the model
knn = mlaKNNClassifier(k=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)


def run_benchmark(models, model_names, benchmark="OpenML-CC18"):
    results = pd.DataFrame(columns=["dataset", "model", "score"])  # Create DataFrame for results
    benchmark_suite = openml.study.get_suite(benchmark)  # Obtain the benchmark suite

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
    subset_benchmark_suite = benchmark_suite.tasks[:10]

    # Iterate over the subset of tasks
    for task_id in subset_benchmark_suite:
        task = openml.tasks.get_task(task_id)  # Download the OpenML task
        features, targets = task.get_X_and_y()  # Get the data
        for model_idx in range(len(models)):  # Iterate over all models
            score = np.mean(cross_val_score(models[model_idx], features, targets, cv=10, scoring="roc_auc_ovr"))
            model_name = model_names[model_idx] if model_names else str(models[model_idx])
            results = pd.concat([results, pd.DataFrame([[task_id, model_name, score]], columns=results.columns)],
                                ignore_index=True)
    results.to_csv("results.csv", index=False)


# Define pipelines
KNN = make_pipeline(SimpleImputer(strategy='constant'), StandardScaler(), KNeighborsClassifier())
KNN2 = make_pipeline(SimpleImputer(strategy='constant'), StandardScaler(), weiKNNClassifier())
models = [KNN, KNN2]
model_names = ["KNN", "KNN2"]

# Run the benchmark
run_benchmark(models=models, model_names=model_names)

# Load results and calculate average rank
results = pd.read_csv("results.csv")
avg_rank = results.groupby('dataset').score.rank(pct=True).groupby(results.model).mean()

print("avg rank: ", avg_rank)

# Calculate accuracy
acc = accuracy(y_test, y_pred)
print("Accuracy: ", acc)


# Plot cross-validation results
def plot_cv(results_cv, metric='Accuracy'):
    fig, ax = plt.subplots()
    ax.boxplot(results_cv)
    ax.set_xticklabels(results_cv.columns)
    ax.set_ylabel(metric)
    ax.set_title('Cross-validation results for KNN and KNN2 in a dataset')
    plt.show()


# Calculate F1 score
f1 = f1_score(y_test, y_pred, average='weighted')
print("F1 Score: ", f1)

# Calculate and print precision, recall, and F1 score for each class
precision = precision_score(y_test, y_pred, average=None)
recall = recall_score(y_test, y_pred, average=None)
f1_per_class = 2 * (precision * recall) / (precision + recall)
print("Precision per class: ", precision)
print("Recall per class: ", recall)
print("F1 Score per class: ", f1_per_class)
