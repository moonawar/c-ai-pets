import pandas as pd

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
        for col in source.columns:
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
        print(self.data)
        pass

    def _rescale(self, data : pd.Series, min : float, max : float) -> pd.Series:
        # rescale data to 0 - 100
        return (data - min) / (max - min) * 100
    
    def saveTo(cls, filename : str):
        pass

    @staticmethod
    def loadFrom(cls, filename : str) -> 'KnnModel':
        pass

if __name__ == "__main__":
    model = KnnModel(3)
    model.train('data/data_train.csv')

