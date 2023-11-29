import sys
sys.path.append('..')
import numpy as np
from sklearn.metrics import accuracy_score
from model import Model

class NaiveBayes(Model):
    def __init__(self):
        self.classes = None
        self.prior_probs = None
        self.feature_stats_by_class = None
        self.predictions = None

    def fit(self, X, y):
        self.classes = y.value_counts().index.tolist()
        total_samples = len(y)
        self.prior_probs = {c: y.value_counts()[c] / total_samples for c in self.classes}

        self.feature_stats_by_class = {}
        for c in self.classes:
            subset = X[y == c]
            stats = {'mean': subset.mean(), 'std': subset.std()}
            self.feature_stats_by_class[c] = stats

    def gaussian_prob(self, x, mean, std):
        exponent = np.exp(-((x - mean) ** 2) / (2 * (std ** 2)))
        return (1 / (np.sqrt(2 * np.pi) * std)) * exponent

    def predict_one(self, sample):
        posteriors = {}
        for c in self.classes:
            likelihood = 1
            for feature, value in sample.items():
                mean = self.feature_stats_by_class[c]['mean'][feature]
                std = self.feature_stats_by_class[c]['std'][feature]
                likelihood *= self.gaussian_prob(value, mean, std)

            posteriors[c] = likelihood * self.prior_probs[c]

        return max(posteriors, key=posteriors.get)

    def predict(self, samples):
        self.predictions = []
        for _, sample in samples.iterrows():
            prediction = self.predict_one(sample)
            self.predictions.append(prediction)
        return self.predictions

    def sklearn_accuracy(self, y_true):
        acc = accuracy_score(y_true, self.predictions)
        print("Akurasi model Naive Bayes (dengan sklearn): {:.2f}%".format(acc * 100))
        return accuracy_score(y_true, self.predictions)

    def manual_accuracy(self, y_true):
        correct_predictions = sum(true_label == predicted_label for true_label, predicted_label in zip(y_true, self.predictions))
        acc = correct_predictions / len(y_true)
        print("Akurasi model Naive Bayes (tanpa sklearn): {:.2f}%".format(acc * 100))
        return acc

    def main():
        # tanya user untuk pakai model atau tidak
        user_input_correct = False
        model_loaded = False
        while not user_input_correct:
            print("Apakah ingin menggunakan model?")
            print(" 1 : ya")
            print(" 2 : tidak")
            with_model = int(input())

            if (with_model == 1):
                user_input_correct = True
                model_loaded = True
                model = NaiveBayes.loadFile("../models/naive_bayes_model.pkl")
            elif (with_model == 2):
                user_input_correct = True
                model = NaiveBayes()
            else:
                print("Input tidak valid?")
            

        data = model.load()
        # Memisahkan fitur (features) dan label (target)
        x = data.drop('price_range', axis=1)
        y = data['price_range']
        # print(X)

        # Inisialisasi dan melatih model NaiveBayes
        if not model_loaded:
            model.fit(x, y)
            
        model.predict(x)
        model.sklearn_accuracy(y)
        model.manual_accuracy(y)

        # tanya user untuk menyimpan model atau tidak
        user_input_correct = False
        while not user_input_correct:
            print("Apakah ingin menyimpan model?")
            print(" 1 : ya")
            print(" 2 : tidak")
            user_want_to_save = int(input())

            if (user_want_to_save == 1):
                user_input_correct = True
                model.save('../models/naive_bayes_model.pkl')
            elif (user_want_to_save == 2):
                user_input_correct = True
                continue
            else:
                print("Input tidak valid?")
                

if __name__ == "__main__":
    NaiveBayes.main()
