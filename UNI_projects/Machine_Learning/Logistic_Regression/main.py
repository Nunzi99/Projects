import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# da finire domani
class logistic_regression:
    # Costruttore vuoto
    def __init__(self):
        self.alpha = 0.1
        self.num_iterations = 50000
        costs = []

    def hypothesis(self, X, Theta):
        dot_product = np.dot(X, Theta)
        hypothesis = 1 / (1 + np.exp(-dot_product))
        return hypothesis
    
    def cost_function(self, hypothesis, y):
        m = len(y)
        # genero un piccolo valore da aggiungere all'ipotesi, ciò ci permetterà di evitare problemi con il log nel caso l'ipotesi fosse 0
        epsilon = 1e-15
        sum = np.sum(y * np.log(hypothesis + epsilon) + (1 - y) * np.log(1 - hypothesis + epsilon))
        cost = -(1 / m) * sum
        return cost
    
    def gradient_descent(self, X, y, Theta, alpha, num_iterations):
        m = len(y)
        # inizializzo il vettore dei costi
        costs = []
        for i in range(num_iterations):
            # Calcola la probabilità
            hypothesis = self.hypothesis(X, Theta)
            # Calcola il vettore degli errori
            error = hypothesis - y  
            # Calcola il gradiente,è necessario trasporre X per poter fare la moltiplicazione scalare con il vettore degli errori
            gradient = (1 / m) * np.dot(X.T, error)
            # Calcolo il costo e lo aggiungo al vettore dei costi
            cost = self.cost_function(hypothesis, y)
            # Termina se il costo smette di diminuire
            if len(costs) > 0 and costs[-1] - cost < 1e-5:
                break
            costs.append(cost)
            # Aggiorna i pesi
            Theta -= alpha * gradient
        return Theta, costs
    
    def train(self, X_train, y_train):
        self.Theta = np.zeros(X_train.shape[1])
        self.Theta, self.costs = self.gradient_descent(X_train, y_train, self.Theta, self.alpha, self.num_iterations)

    def predict(self, X_test, y_test):
        probabilities = self.hypothesis(X_test, self.Theta)
        cost_test = self.cost_function(probabilities, y_test)
        # creiamo il vettore di predizioni
        predictions = []
        # per ogni predizione 
        for p in probabilities:
            # se la predizione é maggiore o uguale a 0.5 allora aggiungiamo 1 al vettore di predizioni altrimenti 0
            if p >= 0.5:
                predictions.append(1)   
            else:
                predictions.append(0)
        return predictions, cost_test

# definisco la funzione per il calcolo delle metriche di Cohen da un dataframe che rappresenta una matrice di confusione
def metrics_calculator(confusion_matrix):
    # converto la matrice di confusione in un array numpy
    confusion_matrix = np.array(confusion_matrix)
    tp = confusion_matrix[0][0]
    tn = confusion_matrix[1][1]
    fp = confusion_matrix[0][1]
    fn = confusion_matrix[1][0]
    m = tp + tn + fp + fn
    recall = tp / (tp + fn)
    specificity = tn / (tn + fp)
    accuracy = (tp + tn) / m

    # calcolo il coefficiente Kappa di Cohen k = (accuracy - pe) / (1 - pe)

    p_plus = ((tp + fn) / m) * ((tp + fp) / m)
    p_minus = ((tn + fp) / m) * ((tn + fn) / m)

    pe = (p_plus + p_minus)

    k_cohen = (accuracy - pe) / (1 - pe)

    return recall, specificity, accuracy, k_cohen

# leggo il dataset
dataframe = pd.read_csv('wdbc_selection.csv')
dataframe = dataframe.sample(frac=1)
# preprocessamento dei dati
# normalizzazione già eseguita con weka
# genero la matrice di features X e il vettore di obiettivi y

y = dataframe['Diagnosis'].values
# convertiamo i valori della 'Diagnosis' in 'y' in '0' e '1', con 1 per 'M' e 0 per 'B'
y = np.where(y == 'M', 1, 0)
X = dataframe.drop('Diagnosis', axis=1).values
# creo il bias e lo aggiungo in testa a X
bias = np.ones((X.shape[0], 1))
X = np.hstack((bias, X))

# divido il dataset in training (70%) e test (30%)
train_size = int(len(X) * 0.70)
X_train = X[:train_size]
y_train = y[:train_size]
X_test = X[train_size:]
y_test = y[train_size:]

logistic_regression = logistic_regression()
logistic_regression.train(X_train, y_train)
predictions = logistic_regression.predict(X_test, y_test)
costs = logistic_regression.costs


# confronto con le previsioni ottenute con sklearn
Theta_sk = np.zeros(X.shape[1])
clf = linear_model.LogisticRegression()
clf.fit(X_train, y_train)
predictions_sk = clf.predict(X_test)

# ottengo le matrici di confusione sia per la mia implementazione che per sklearn, [[1, 0], :][:, [1, 0]] serve per invertire le righe e le colonne
custom_confusion_matrix = confusion_matrix(y_test, predictions[0])[[1, 0], :][:, [1, 0]]
custom_confusion_matrix = pd.DataFrame(custom_confusion_matrix, columns=['M','B'], index=['M','B'])
sklearn_confusion_matrix = confusion_matrix(y_test, predictions_sk)[[1, 0], :][:, [1, 0]]
sklearn_confusion_matrix = pd.DataFrame(sklearn_confusion_matrix, columns=['M','B'], index=['M','B'])
weka_confusion_matrix = pd.DataFrame([[73, 1], [1, 118]], columns=['M','B'], index=['M','B'])
# stampo le matrici di confusione
print("Matrici di confusione implementata:")
print(custom_confusion_matrix)
print("Matrici di confusione sklearn:")
print(sklearn_confusion_matrix)

# ottengo le metriche per le due matrici di confusione e le inserisco in due array numpy

recall, specificity, accuracy, k_cohen = metrics_calculator(custom_confusion_matrix)
metrics_custom = np.array([recall, specificity,  accuracy, k_cohen])

recall_sk, specificity_sk,  accuracy_sk, k_cohen_sk = metrics_calculator(sklearn_confusion_matrix)
metrics_sklearn = np.array([recall_sk, specificity_sk,  accuracy_sk, k_cohen_sk])

recall_w, specificity_w, accuracy_w, k_cohen_w = metrics_calculator(weka_confusion_matrix)
metrics_w = np.array([recall_w, specificity_w,  accuracy_w, k_cohen_w])

# dai array numpy creo un dataframe con le metriche e stampo il dataframe
metrics = np.vstack((metrics_custom, metrics_sklearn, metrics_w))
columns = ['Recall', 'Specificity', 'Accuracy', 'K-Cohen']
metrics = pd.DataFrame(metrics, columns=columns)
metrics.index = ['Implementata', 'Sklearn', 'Weka']
print(metrics)


# Stampo il grafico del costo
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(costs)
plt.title('Evoluzione Costo')
plt.xlabel('Iterazioni')
plt.ylabel('Costo')

# Stampo le metriche
plt.subplot(1, 2, 2)
plt.axis('off')

text = 'METRICHE:\n\n'
text += f'Loss Train: {round(costs[-1],2)}\n'
text += f'Loss Test: {round(predictions[1],2)}\n'
text += f'Matrice di confusione:\n{custom_confusion_matrix}\n'
text += f'Matrice di confusione sklearn:\n{sklearn_confusion_matrix}\n'
text += f'Matrice di confusione weka:\n{weka_confusion_matrix}\n'
text += f'Recall implementata: {round(recall,2)} vs sklearn: {round(recall_sk,2)} vs weka: {round(recall_w,2)}\n'
text += f'Specificity implementata: {round(specificity,2)} vs sklearn: {round(specificity_sk,2)} vs weka: {round(specificity_w,2)}\n'
text += f'Accuracy implementata: {round(accuracy,2)} vs sklearn: {round(accuracy_sk,2)} vs weka: {round(accuracy_w,2)}\n'
text += f'K-Cohen implementata: {round(k_cohen,2)} vs sklearn: {round(k_cohen_sk,2)} vs weka: {round(k_cohen_w,2)}\n'

plt.text(0.2, 0, text)
plt.show()
