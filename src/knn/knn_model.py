import sys
sys.path.append('..')
import pandas as pd
import numpy as np
from model import Model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

class KNN(Model):
    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predictions = []
        for x in X:
            prediction = self._predict(x)
            predictions.append(prediction)
        return np.array(predictions)

    def _predict(self, x):
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        predicted = max(set(k_nearest_labels), key=k_nearest_labels.count)
        
        return predicted
    
    def main():
        # tanya user untuk pakai model atau tidak
        user_input_correct = False
        while not user_input_correct:
            print("Apakah ingin menggunakan model?")
            print(" 1 : ya")
            print(" 2 : tidak")
            with_model = int(input())

            if (with_model == 1):
                user_input_correct = True
                knn = KNN.loadFile("../models/knn_model.pkl")
            elif (with_model == 2):
                user_input_correct = True
                k = 17
                knn = KNN(k)
                knn_sklearn = KNeighborsClassifier(n_neighbors=k)
                data_train_url = 'https://drive.google.com/file/d/1cK8X26xG3eei3UI7Vdy_ocpDUT3Yhd9f/view?usp=sharing'
                file_id = data_train_url.split('/')[-2]
                data_train = pd.read_csv(f'https://drive.google.com/uc?id={file_id}')

                X_train = data_train.drop('price_range', axis=1).values
                y_train = data_train['price_range'].values

                knn.fit(X_train, y_train)
                knn_sklearn.fit(X_train, y_train)
            else:
                print("Input tidak valid?")
        

        data_validation_url = 'https://drive.google.com/file/d/1YsrrsvceH2Fqs0Ke-LaNbeA6ITD50E1h/view?usp=sharing'
        file_id = data_validation_url.split('/')[-2]
        data_validation = pd.read_csv(f'https://drive.google.com/uc?id={file_id}')


        X_validation = data_validation.drop('price_range', axis=1).values
        y_validation = data_validation['price_range'].values

        y_pred = knn.predict(X_validation)
        y_pred_sklearn = knn_sklearn.predict(X_validation)

        accuracy_value = accuracy(y_validation, y_pred)
        accuracy_value_sklearn = accuracy_score(y_validation, y_pred_sklearn)
        
        precision_value = precision(y_validation, y_pred)
        # precision_value_sklearn = precision_score(y_validation, y_pred_sklearn, average='macro')
        
        recall_value = recall(y_validation, y_pred)
        # recall_value_sklearn = recall_score(y_validation, y_pred_sklearn)
        
        print("Akurasi model KNN: {:.2f}%".format(accuracy_value * 100))
        print("Akurasi model KNN sklearn: {:.2f}%".format(accuracy_value_sklearn * 100))
        
        print("Precision model KNN: {:.2f}%".format(precision_value * 100))
        # print("Precision model KNN sklearn: {:.2f}%".format(precision_value_sklearn * 100))
        
        print("Recall model KNN: {:.2f}%".format(recall_value * 100))
        # print("Recall model KNN sklearn: {:.2f}%".format(recall_value_sklearn * 100))
        
        # tanya user untuk menyimpan model atau tidak
        user_input_correct = False
        while not user_input_correct:
            print("Apakah ingin menyimpan model?")
            print(" 1 : ya")
            print(" 2 : tidak")
            user_want_to_save = int(input())

            if (user_want_to_save == 1):
                user_input_correct = True
                knn.save('../models/knn_model.pkl')
            elif (user_want_to_save == 2):
                user_input_correct = True
                continue
            else:
                print("Input tidak valid?")


def euclidean_distance(x1, x2):
    distance = 0
    for i in range(len(x1)):
        distance += (x1[i] - x2[i]) ** 2
    return np.sqrt(distance)

    
def accuracy(y_true, y_pred):
    correct = 0
    for yt, yp in zip(y_true, y_pred):
        if yt == yp:
            correct += 1
    return correct / len(y_true)

def precision(y_true, y_pred):
    true_positive = 0
    predicted_positive = 0
    for yt, yp in zip(y_true, y_pred):
        if yp == 1:
            predicted_positive += 1
            if yt == yp:
                true_positive += 1
    return true_positive / predicted_positive

def recall(y_true, y_pred):
    true_positive = 0
    actual_positive = 0
    for yt, yp in zip(y_true, y_pred):
        if yt == 1:
            actual_positive += 1
            if yt == yp:
                true_positive += 1
    return true_positive / actual_positive

if __name__ == "__main__":
    KNN.main()