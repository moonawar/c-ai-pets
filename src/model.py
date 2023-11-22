from abc import ABC, abstractmethod

class Model(ABC):
    @abstractmethod
    def save(self, filename : str) : bool:
        '''
            Save the model to a file
            will be saved to path: models/[filename]
        
            Args:
                filename (str): the name of the file to save the model to
            Returns:
                bool: True if the model was saved successfully, False otherwise
        '''
        pass

    @abstractmethod
    def load(self, filename):
        '''
            Load the model from a file
            will be loaded from path: models/[filename]
        
            Args:
                filename (str): the name of the file to load the model from
            Returns:
                bool: True if the model was loaded successfully, False otherwise
        '''
        pass