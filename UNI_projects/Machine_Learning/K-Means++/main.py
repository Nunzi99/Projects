# aggiustare e fare in modo che ad ogni famiglia assegni un cluster

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

class KMeansPlusPlus:
    def __init__(self, k, max_iterations):
        self.k = k
        self.max_iterations = max_iterations
        self.centroids = None
        self.train_losses = []
        self.validation_losses = []
        pass
    # definisco il metodo per computare le distanze con il centroide più vicino
    def minimum_distance(self, dataframe):
        # per ogni elemento nel dataframe aggiungo la distanza con il centroide più vicino
        distances = np.array([min(np.linalg.norm(element - centroid) ** 2 for centroid in self.centroids) for element in dataframe])
        return distances
    # definisco il metodo per computare la matrice delle distanze con tutti i centroidi
    def distance_matrix(self, dataframe):
        distances = np.array([[np.linalg.norm(point - centroid) for centroid in self.centroids] for point in dataframe])
        return distances
    
    # definisco il metodo per generare k centroidi con il kmeans ++
    def centroids_initialization(self, dataframe):
        # Genero i centroidi usando la tecnica k-means++
        # Seleziono uno dei punti come primo centroide
        first_centroid = dataframe[np.random.choice(dataframe.shape[0])]
        # Aggiungo il primo centroide alla lista dei centroidi
        self.centroids = [first_centroid]
        # Per i k-1 centroidi rimanenti
        for i in range(self.k - 1):
            # Per ogni elemento nel dataframe calcolo la distanza con il centroide più vicino
            distances = self.minimum_distance(dataframe)
            # Calcolo la probabilità di un punto di essere scelto come prossimo centroide in base a quanto esso è distante dal centroide più vicino
            probabilities = distances / distances.sum()
            # Seleziono il nuovo centroide usando come probabilità di essere selezionato quella calcolata nel passo precedente
            new_centroid_index = np.random.choice(dataframe.shape[0], p=probabilities)
            # Aggiungo il nuovo centroide alla lista dei centroidi
            new_centroid = dataframe[new_centroid_index]
            self.centroids.append(new_centroid)
        return self.centroids
    # definisco il metodo per addestrare il modello
    def fit(self, train_set_X, validation_set_X):
        # Genero i centroidi usando la tecnica k-means++
        self.centroids_initialization(train_set_X)
        # Eseguiamo il clustering
        for i in range(self.max_iterations):
            # Calcolo la matrice delle distanze con tutti i centroidi
            distances = self.distance_matrix(train_set_X)
            # Ottengo l'indice del centroide più vicino per ogni punto in un 1D array
            nearest_centroid_index = np.argmin(distances, axis=1)
            # Creo un array per i nuovi centroidi
            new_centroids = np.zeros_like(self.centroids)
            # Per ogni centroide
            for j in range(self.k):
                # Seleziono dal dataframe i punti che lo hanno come centroide più vicino
                assigned_points = train_set_X[nearest_centroid_index == j]
                # considero il caso in cui ad un centroide non ci siano punti assegnati, in tal caso lo si riassegna
                if len(assigned_points) == 0:
                    new_centroids[j] = train_set_X[np.random.choice(train_set_X.shape[0])]
                # aggiungo ai nuovi centroidi un centroide che è la media dei punti assegnati al centroide attuale
                new_centroids[j] = np.mean(assigned_points, axis=0)
            # Se i centroidi non sono cambiati termino perchè si è raggiunta la convergenza
            if np.array_equal(self.centroids, new_centroids):
                break
            # Aggiorno i centroidi
            self.centroids = new_centroids
            # Calcolo la loss per il train set
            loss = np.sum(self.minimum_distance(train_set_X))
            # Aggiorno la loss
            self.train_losses.append(loss)
            self.validation_losses.append(np.sum(self.minimum_distance(validation_set_X)))
    # Definisco il metodo per creare la matrice di confusione
    def create_confusion_matrix(self, dataframe, labels):
        # calcolo la matrice delle distanze con i centroidi (riga = elementi, colonna = distanza con il relativo centroide)
        distances = self.distance_matrix(dataframe)
        # ottengo l'indice del centroide maggiormente vicino per ogni punto in un 1D array
        nearest_centroid_index = np.argmin(distances, axis=1)
        # unisco i 2 array
        centroid_family_couples = np.column_stack([nearest_centroid_index, labels])
        # creo la matrice 
        labels = np.unique(labels)
        centroids_indexes = np.unique(nearest_centroid_index)
        confusion_matrix = pd.DataFrame(0, index=labels, columns=centroids_indexes)
        # Per ogni coppia cluster-famiglia
        for cluster, family in centroid_family_couples:
            # Incrementa di 1 il valore del relativo cluster nella riga relativa alla famiglia
            confusion_matrix.loc[family, cluster] += 1
        return confusion_matrix
    
    def get_metrics(self, dataframe, labels):
        confusion_matrix = self.create_confusion_matrix(dataframe, labels)
        loss = np.sum(self.minimum_distance(dataframe))
        # creo un dizionario che salverà i valori come key e una lista contenente famiglia e cluster come value
        value_family_cluster = {}
        # ottengo tutti i valori contenuti nella matrice di confusione e li ordino in ordine decrescente
        values = np.sort(confusion_matrix.values.flatten())[::-1]
        # ottengo il numero totale di valori presenti nella matrice di confusione
        total_values = np.sum(confusion_matrix.values)
        # popolo il dizionario
        # itero sulle righe/famiglie
        for index, family in confusion_matrix.iterrows():
            # itero sulle colonne/cluster
            for cluster in confusion_matrix:
                value = family[cluster].item()
                value_family_cluster[value] = [index, cluster]
        # creo una variabile che salverà quanti valori sono correttamente assegnati
        correctly_assigned = 0
        # creo un dizionario che salverà i valori come key e una lista contenente le famiglie e il relativo cluster assegnato
        assignnments = {}
        # itero sulla lista ordinata dei valori
        for value in values:
            # ottengo la famiglia e il cluster assegnato
            family, cluster = value_family_cluster[value]
            # se la famiglia e il cluster non sono gia presenti nel dizionario lo aggiungo
            if family not in assignnments and cluster not in assignnments.values():
                assignnments[family] = cluster
                correctly_assigned += value
        # calcolo la precisione
        correctly_assigned_instances = correctly_assigned / total_values
        return confusion_matrix, loss, assignnments, correctly_assigned_instances
    def predict(self, dataframe):
        distances = self.distance_matrix(dataframe)
        nearest_centroid_index = np.argmin(distances, axis=1)
        return nearest_centroid_index
# leggo il dataset
dataframe = pd.read_csv("Frogs_MFCCs_processed.csv")
# mescolo il dataset
dataframe = dataframe.sample(frac=1).reset_index(drop=True)
# divido in train (60%), validation (15%) e test (15%)
train_set = dataframe.iloc[:int(dataframe.shape[0] * 0.6)]
validation_set = dataframe.iloc[int(dataframe.shape[0] * 0.6):int(dataframe.shape[0] * 0.75)]
test_set = dataframe.iloc[int(dataframe.shape[0] * 0.75):]

# divido i dataset tra X e y (famiglie) e li trasformo in array numpy
train_set_X = np.array(train_set.iloc[:, :-1])
train_set_y = np.array(train_set.iloc[:, -1])

validation_set_X = np.array(validation_set.iloc[:, :-1])
validation_set_y = np.array(validation_set.iloc[:, -1])

test_set_X = np.array(test_set.iloc[:, :-1])
test_set_y = np.array(test_set.iloc[:, -1])

# Desidero ottenere 4 cluster dato che ci sono 4 famiglie
k = 4

kmeans = KMeansPlusPlus(k, 100)
kmeans.fit(train_set_X, validation_set_X)

confusion_matrix_train, loss_train, family_cluster_train, correctly_assigned_instances_train = kmeans.get_metrics(train_set_X, train_set_y)
confusion_matrix_validation, loss_validation, family_cluster_validation, correctly_assigned_instances_validation = kmeans.get_metrics(validation_set_X, validation_set_y)
confusion_matrix_test, loss_test, family_cluster_test, correctly_assigned_instances_test = kmeans.get_metrics(test_set_X, test_set_y)

silhouette_score_train = silhouette_score(train_set_X, kmeans.predict(train_set_X))
silhouette_score_validation = silhouette_score(validation_set_X, kmeans.predict(validation_set_X))
silhouette_score_test = silhouette_score(test_set_X, kmeans.predict(test_set_X))

print(f'Confusion Matrix Train:\n{confusion_matrix_train}')
print(f'The cluster-family assignments are: {family_cluster_train} and {round(correctly_assigned_instances_train * 100,2)}% of the instances were correctly assigned')

print(f'Confusion Matrix Validation:\n{confusion_matrix_validation}')
print(f'The cluster-family assignments are: {family_cluster_validation} and {round(correctly_assigned_instances_validation * 100,2)}% of the instances were correctly assigned')

print(f'Confusion Matrix Test:\n{confusion_matrix_test}')
print(f'The cluster-family assignments are: {family_cluster_test} and {round(correctly_assigned_instances_test * 100,2)}% of the instances were correctly assigned')

# Implemento il modello con scikitlearn
kmeans_skt = KMeans(n_clusters=k)
kmeans_skt.fit(train_set_X)

skt_silhouette_score_train = silhouette_score(train_set_X, kmeans_skt.predict(train_set_X))
skt_silhouette_score_validation = silhouette_score(validation_set_X, kmeans_skt.predict(validation_set_X))
skt_silhouette_score_test = silhouette_score(test_set_X, kmeans_skt.predict(test_set_X))

skt_loss_test = kmeans_skt.inertia_

# Stampo il grafico
plt.figure(figsize=(12, 5))

plt.subplot(1, 3, 1)
plt.title('Loss Train')
plt.plot(kmeans.train_losses)
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.subplot(1, 3, 2)
plt.title('Loss Validation')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.plot(kmeans.validation_losses)
plt.subplot(1, 3, 3)
plt.axis('off')

plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.4, hspace=0.4)

text = 'METRICHE:\n\n'
# Stampo le accuratezze
text += f'Accuracy Train: {round(correctly_assigned_instances_train,2)}\n'
text += f'Accuracy Validation: {round(correctly_assigned_instances_validation,2)}\n'
text += f'Accuracy Test: {round(correctly_assigned_instances_test,2)}\n'
# Stampo le silhouette
text += f'Silhouette Score Train: {round(silhouette_score_train,2)}\n'
text += f'Silhouette Score Validation: {round(silhouette_score_validation,2)}\n'
text += f'Silhouette Score Test: {round(silhouette_score_test,2)}\n'
text += f'Silhouette Score Sklearn Test: {round(skt_silhouette_score_test,2)}\n'
# Stampo la loss di train
text += f'Loss Test: {round(loss_train,2)}\n'
text += f'Loss Sklearn Test: {round(skt_loss_test,2)}\n'
plt.text(0.2, 0, text)
plt.show()