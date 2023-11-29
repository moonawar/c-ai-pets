import sys
sys.path.append('..')
import pandas as pd
import numpy as np
from model import Model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

class Predict(Model):
    def __init__(self):
        self.k = 19

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
        data_train = pd.read_csv('data/data_train.csv')
        data_test = pd.read_csv('data/test.csv')

        # untuk training
        X_train = data_train.drop('price_range', axis=1).values
        y_train = data_train['price_range'].values

        # untuk prediksi
        X_test = data_test.values

        # tanya user untuk pakai model atau tidak
        user_input_correct = False
        while not user_input_correct:
            print("Apakah ingin menggunakan model?")
            print(" 1 : ya")
            print(" 2 : tidak")
            with_model = int(input())

            if (with_model == 1):
                user_input_correct = True
                model = Predict.loadFile("../models/predict.pkl")
            elif (with_model == 2):
                user_input_correct = True
                model = Predict()
                model.fit(X_train, y_train)
            else:
                print("Input tidak valid?")

        y_pred = model.predict(X_test)
        data_test['price_range'] = y_pred
        predicted_results = data_test[['id', 'price_range']]
        predicted_results.to_csv('data/submission.csv', index=False)
        
        # tanya user untuk menyimpan model atau tidak
        user_input_correct = False
        while not user_input_correct:
            print("Apakah ingin menyimpan model?")
            print(" 1 : ya")
            print(" 2 : tidak")
            user_want_to_save = int(input())

            if (user_want_to_save == 1):
                user_input_correct = True
                model.save('../models/predict.pkl')
            elif (user_want_to_save == 2):
                user_input_correct = True
                continue
            else:
                print("Input tidak valid?")

def euclidean_distance(x1, x2):
    distance = 0
    for i in range(len(x1)-1):
        distance += (x1[i] - x2[i]) ** 2
    return np.sqrt(distance)

if __name__ == "__main__":
    Predict.main()