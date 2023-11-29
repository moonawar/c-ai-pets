from knn.knn_model import KNN
from naive_bayes.naivebayes_model import NaiveBayes

if __name__ == "__main__":
    input_correct = False
    while not input_correct:
        print("Pilih algoritma:\n 1 = KNN\n 2 = Naive bayes")
        algorithm = int(input())
        if algorithm == 1:
            input_correct = True
            KNN.main()
        elif algorithm == 2:
            input_correct = True
            NaiveBayes.main()
        else:
            print("Masukan salah!")