import sys
sys.path.append('..')
import pandas as pd
import numpy as np
from model import Model

class KNN(Model):
    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    def _predict(self, x):
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        common = np.argmax(np.bincount(k_nearest_labels))
        return common
    
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
                data_train_url = 'https://drive.google.com/file/d/1cK8X26xG3eei3UI7Vdy_ocpDUT3Yhd9f/view?usp=sharing'
                file_id = data_train_url.split('/')[-2]
                data_train = pd.read_csv(f'https://drive.google.com/uc?id={file_id}')

                X_train = data_train.iloc[:, :-1].values
                y_train = data_train.iloc[:, -1].values

                knn.fit(X_train, y_train)
            else:
                print("Input tidak valid?")
        

        data_validation_url = 'https://drive.google.com/file/d/1YsrrsvceH2Fqs0Ke-LaNbeA6ITD50E1h/view?usp=sharing'
        file_id = data_validation_url.split('/')[-2]
        data_validation = pd.read_csv(f'https://drive.google.com/uc?id={file_id}')

        X_validation = data_validation.iloc[:, :-1].values
        y_validation = data_validation.iloc[:, -1].values

        y_pred = knn.predict(X_validation)

        accuracy_value = accuracy(y_validation, y_pred)
        print("Akurasi model KNN: {:.2f}%".format(accuracy_value * 100))
        
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
    return np.sqrt(np.sum((x1 - x2) ** 2))

def accuracy(y_true, y_pred):
    correct = 0
    for yt, yp in zip(y_true, y_pred):
        if yt == yp:
            correct += 1
    return correct / len(y_true)

if __name__ == "__main__":
    KNN.main()