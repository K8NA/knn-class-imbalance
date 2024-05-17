import numpy as np
from scipy.spatial.distance import euclidean
from collections import Counter

from mla.base import BaseEstimator

class KNNBase(BaseEstimator):
    def __init__(self, k=5, distance_func=euclidean):
        """Base class for Nearest neighbors classifier and regressor.

        Parameters
        ----------
        k : int, default 5
            The number of neighbors to take into account. If 0, all the
            training examples are used.
        distance_func : function, default euclidean distance
            A distance function taking two arguments. Any function from
            scipy.spatial.distance will do.
        """

        self.k = None if k == 0 else k  # l[:None] returns the whole list
        self.distance_func = distance_func

    def aggregate(self, neighbors_targets, distances):
        raise NotImplementedError()

    def _predict(self, X=None):
        predictions = [self._predict_x(x) for x in X]

        return np.array(predictions)

    def _predict_x(self, x):
        """Predict the label of a single instance x."""

        # Explicação:
        # Primeiro, calcula as distâncias entre x e todas as instâncias no conjunto de treinamento.
        # Em seguida, ordena essas instâncias pelo valor da distância e seleciona os k vizinhos mais próximos. 
        # Finalmente, chama o método aggregate para combinar os rótulos (ou valores alvo) dos vizinhos e fazer a previsão final.

        # calcular distâncias entre x e todos os exemplos no conjunto de treinamento
        distances = [self.distance_func(x, example) for example in self.X]

        # ordenar todos os exemplos pela sua distância até x e manter o valor alvo
        neighbors = sorted(zip(distances, self.y), key=lambda x: x[0])

        # obter alvos do k-vizinho mais próximo e agregá-los (o mais comum ou média)
        neighbors_targets = [target for (_, target) in neighbors[: self.k]]
        neighbors_distances = [dist for (dist, _) in neighbors[: self.k]]

        return self.aggregate(neighbors_targets, neighbors_distances)


class KNNClassifier(KNNBase):
    """Nearest neighbors classifier.

    Note: if there is a tie for the most common label among the neighbors, then
    the predicted label is arbitrary."""

    # Explicação2:
    # pondera os rótulos dos vizinhos pelo inverso das suas distâncias
    # utiliza um Counter para manter a contagem ponderada dos rótulos
    # retorna o rótulo mais comum, tendo em conta esses pesos
    # fornece uma previsão robusta, favorecendo rótulos de vizinhos mais próximos ao ponto de consulta

    def aggregate(self, neighbors_targets, distances):
        """Return the most common target label weighted by inverse distance."""

        weighted_labels = Counter()
        for label, distance in zip(neighbors_targets, distances):
            if distance != 0: # evitar divisão por 0
                weighted_labels[label] += 1 / distance

        if not weighted_labels:  # se o Counter for vazio
            return neighbors_targets[0]  # selecionar aleatoriamente o primeiro alvo
        else:
            return weighted_labels.most_common(1)[0][0]


class KNNRegressor(KNNBase):
    """Nearest neighbors regressor."""

    # Explicação3:
    # calcula a média dos valores alvo dos vizinhos mais próximos, ponderando-os pelo inverso de suas distâncias, 
    # garantindo que não ocorra divisão por zero ao retornar a média simples se todas as distâncias forem zero.


    def aggregate(self, neighbors_targets, distances):
        """Return the mean of all targets weighted by inverse distance."""

        weighted_sum = 0
        total_weight = 0
        for target, distance in zip(neighbors_targets, distances):
            if distance != 0:  # evitar divisão por 0
                weighted_sum += target / distance
                total_weight += 1 / distance

        if total_weight == 0:  # se todas as distâncias forem 0
            return np.mean(neighbors_targets)  # retorna a média sem ponderação
        else:
            return weighted_sum / total_weight

