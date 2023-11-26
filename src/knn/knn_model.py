import pandas as pd
import numpy as np
class KnnModel:
    k : int
    data : pd.DataFrame
    scaler : [
        {
            'min' : float,
            'max' : float
        }
    ]

    def __init__(self, k) -> None:
        self.k = k
        self.data = pd.DataFrame()
        pass

    def train(self, csv : str):
        # load data from csv
        source = pd.read_csv(csv)
        scaler = []
        target = source.pop('price_range')
        for i, col in enumerate(source.columns):
            # get min and max of each column
            min = source[col].min()
            max = source[col].max()
            scaler.append({
                'min' : min,
                'max' : max
            })

            # rescale column
            source[col] = self._rescale(source[col], min, max)

        # save scaler
        self.scaler = scaler

        source['price_range'] = target

        # rescale data
        self.data = pd.concat([self.data, source])
        pass


    def predict(self, unseen_data : pd.Series) -> int:
        if self.data.empty:
            raise Exception("Model must be trained before predictions can be made")

        # rescale unseen data
        temp = unseen_data.copy()
        for i, col in enumerate(unseen_data.index):
            temp[col] = self._rescale(unseen_data[col], self.scaler[i]['min'], self.scaler[i]['max'])

        unseen_data = temp
        
        # calculate euclidean distance between unseen data and all data in model
        distances = self.data.apply(lambda x: self._euclidean_distance(x[:-1], unseen_data), axis=1)
        distances = distances.sort_values(ascending=True)
        res = []
        for i in range(self.k):
            res.append(self.data.iloc[distances.index[i]]['price_range'])

        res = pd.Series(res)
        modes = res.mode()

        confidence = res.value_counts()[modes[0]] / self.k * 100
        return modes[0]
        

    def _euclidean_distance(self, a : pd.Series, b : pd.Series) -> float:
        # calculate euclidean distance between two rows
        sq = lambda x: x ** 2
        distance = 0
        for i in range(len(a)):
            distance += sq(a.iloc[i] - b.iloc[i])

        return distance ** 0.5

    def _rescale(self, data : pd.Series, min : float, max : float) -> pd.Series:
        # rescale data to 0 - 100
        return (data - min) / (max - min) * 100
    
    def saveTo(cls, filename : str):
        pass

    @staticmethod
    def loadFrom(cls, filename : str) -> 'KnnModel':
        pass

if __name__ == "__main__":
    model = KnnModel(21)
    model.train('data/data_train.csv')

    validation = pd.read_csv('data/data_validation.csv')
    target = validation.pop('price_range')

    correct_count = 0

    for i in range(len(validation)):
        predicted = model.predict(validation.iloc[i])
        actual = target[i]
        if predicted == actual:
            correct_count += 1

    print(f"accuracy: {correct_count / len(validation) * 100}%")