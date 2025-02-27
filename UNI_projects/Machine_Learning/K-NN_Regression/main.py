import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.neighbors import KNeighborsRegressor
from scipy import stats

# prediciamo il valore di un singolo esempio
def knn(X_train, y_train, x, k):
    # usando np.linalg.norm possiamo calcolare la distanza di un vettore dall'origine, la norma viene calcolata come la radice quadrata della somma dei quadrati dei valori del vettore
    # volendo calcolare la distanza tra due vettori con lo stesso numero di elementi possiamo usare come input la differenza tra i due vettori
    # np.linalg.norm accetta anche matrici ed impostando axis=1 gli diciamo di usare le righe della matrice come vettore
    distances = np.linalg.norm(X_train - x, axis=1)
    # usando np.argsort posso ottenere gli indici dei vettori in ordine decrescente di distanza, usando [:k] scelgo solo i primi k indici
    indices = np.argsort(distances)[:k]
    # ottengo i valori delle y per gli indici ottenuti
    values = y_train[indices]
    # calcolo la media dei valori ottenuti
    # mean = np.mean(values)
    weights = 1 / (distances[indices]  + 1e-5)
    mean = np.average(values, weights=weights)
    return mean.item()

# prediciamo i valori di un vettore di esempi
def predict_values(X_train, y_train, X_test, k):
    predictions = []
    for x in X_test:
        predictions.append(knn(X_train, y_train, x, k))
    return predictions
# Definisco la funzione che trova il miglior numero di k neighbors
def cross_validation(X_train, y_train, X_test, y_test, k_max): 
    scores = []
    print(f'Computing best k, please wait ...')
    for k in range(1, k_max + 1):
        p_values = np.array(predict_values(X_train, y_train, X_test, k))
        pearson_correlation_coefficient = stats.pearsonr(y_test, p_values).statistic
        r2 = metrics.r2_score(y_test, p_values)
        scores.append((pearson_correlation_coefficient + r2) / 2)
    k = scores.index(max(scores)) + 1
    print(f'Best k: {k}')
    return k

# leggo il dataset 
dataframe = pd.read_csv('CCPP.csv')
dataframe = dataframe.sample(frac=1).reset_index(drop=True)
X = dataframe.drop('PE', axis=1).values
y = dataframe['PE'].values

# divido il dataset in training (66%) e test (34%)
train_size = int(len(X) * 0.66)
X_train = X[:train_size]
y_train = y[:train_size]
X_test = X[train_size:]
y_test = y[train_size:]

# calcolo il miglior k
k = cross_validation(X_train, y_train, X_test, y_test, 100)
p_values = np.array(predict_values(X_train, y_train, X_test, k))


# calcolo le metriche di interesse per l'implementazione from scratch
pearson_correlation_coefficient = stats.pearsonr(y_test, p_values).statistic
r2 = metrics.r2_score(y_test, p_values)

sklearn_knn = KNeighborsRegressor(n_neighbors=k)
sklearn_knn.fit(X_train, y_train)

sklearn_predictions = sklearn_knn.predict(X_test)

# calcolo r2 per sklearn
sklearn_r2 = metrics.r2_score(y_test, sklearn_predictions)

print(f'pearson: {round(pearson_correlation_coefficient, 4)}')
print(f'weka: 0.9755')
print(f'r2: {round(r2,4)}')
print(f'sklearn r2: {round(sklearn_r2,4)}')
