import pickle
import pandas as pd

class Model():
    def save(self, filename: str):
        try:
            with open(filename, 'wb') as file:
                pickle.dump(self, file)

            print(f"File {filename} saved successfully.")
            return True
        except Exception:
            print(f"Error when saving file {filename}")
            return False

    def load(self):
        data = pd.read_csv('data/data_train.csv')
        return data
    
    def loadFile(filename):
        try:
            with open(filename, 'rb') as file:
                return pickle.load(file)
        except Exception:
            print(f"Error when loading model file {filename}")
            return False