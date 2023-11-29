import sys
sys.path.append('..')
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score
from model import Model
import pandas as pd

class NaiveBayes(Model):
    def __init__(self):
        self.summaries = None

    def fit(self, X, y):
        summaries = {}
        classes = np.unique(y)
        for class_value in classes:
            indices = np.where(y == class_value)
            X_class = X.iloc[indices]
            summaries[class_value] = [(np.mean(attribute), np.std(attribute)) for attribute in X_class.values.T]
        self.summaries = summaries

    def gaussProb(self, x, mean, stdev):
        exponent = np.exp(-((x - mean) ** 2 / (2 * stdev ** 2)))
        return (1 / (np.sqrt(2 * np.pi) * stdev)) * exponent

    def calculate_class_probabilities(self, input_data):
        probabilities = {}
        for class_value, class_summaries in self.summaries.items():
            probabilities[class_value] = 1
            for i in range(len(class_summaries)):
                mean, stdev = class_summaries[i]
                x = input_data[i]
                probabilities[class_value] *= self.gaussProb(x, mean, stdev)
        return probabilities

    def predict(self, input_data):
        probabilities = self.calculate_class_probabilities(input_data)
        best_label, best_prob = None, -1
        for class_value, probability in probabilities.items():
            if best_label is None or probability > best_prob:
                best_prob = probability
                best_label = class_value
        return best_label

    def main():
        data_train = pd.read_csv('data/data_train.csv')
        data_validation = pd.read_csv('data/data_validation.csv')

        # untuk training
        X_train = data_train.drop('price_range', axis=1)
        y_train = data_train['price_range']

        # untuk validasi
        X_validation = data_validation.drop('price_range', axis=1)
        y_validation = data_validation['price_range']

        # tanya user untuk pakai model atau tidak
        user_input = False
        model_loaded = False
        while not user_input:
            print("Apakah ingin menggunakan model?")
            print(" 1 : ya")
            print(" 2 : tidak")
            with_model = int(input())

            if (with_model == 1):
                user_input = True
                model_loaded = True
                model = NaiveBayes.loadFile("../models/naive_bayes_model.pkl")
            elif (with_model == 2):
                user_input = True
                model = NaiveBayes()
                model.fit(X_train, y_train)
            else:
                print("Input tidak valid?")
        
        # HASIL NAIVE-BAYES SCRATCH
        naive_bayes_scratch = [model.predict(x) for x in X_validation.values]
        accuracy_scratch = accuracy_score(y_validation, naive_bayes_scratch)
        precision_scratch = precision_score(y_validation, naive_bayes_scratch, average='macro')
        recall_scratch = recall_score(y_validation, naive_bayes_scratch, average='macro')
        
        print("Hasil model Naive-Bayes: ")
        print("Akurasi model Naive-Bayes: {:.2f}%".format(accuracy_scratch * 100))
        print("Precision model Naive-Bayes: {:.2f}%".format(precision_scratch * 100))
        print("Recall model Naive-Bayes: {:.2f}% \n".format(recall_scratch * 100))

        # HASIL NAIVE-BAYES SCIKIT LEARN
        model_sklearn = GaussianNB()
        model_sklearn.fit(X_train, y_train)
        naive_bayes_sklearn = model_sklearn.predict(X_validation)
        accuracy_sklearn = accuracy_score(y_validation, naive_bayes_sklearn)
        precision_sklearn = precision_score(y_validation, naive_bayes_sklearn, average='macro')
        recall_sklearn = recall_score(y_validation, naive_bayes_sklearn, average='macro')

        print("Hasil model Naive-Bayes sklearn: ")
        print("Akurasi model Naive-Bayes sklearn: {:.2f}%".format(accuracy_sklearn * 100))
        print("Precision model Naive-Bayes sklearn: {:.2f}%".format(precision_sklearn * 100))
        print("Recall model Naive-Bayes sklearn: {:.2f}% \n".format(recall_sklearn * 100))

        # tanya user untuk menyimpan model atau tidak
        user_input = False
        while not user_input:
            print("Apakah ingin menyimpan model?")
            print(" 1 : ya")
            print(" 2 : tidak")
            save = int(input())

            if (save == 1):
                user_input = True
                model.save('../models/naive_bayes_model.pkl')
            elif (save == 2):
                user_input = True
                continue
            else:
                print("Input tidak valid?")

if __name__ == "__main__":
    NaiveBayes.main()
