import pickle
from pathlib import Path

class RNN_Base:
    def LoadParam(self, filepath):
        if filepath.exists():
            with open(filepath, 'rb') as f:
                filedata = pickle.load(f)
                for var in vars(self):
                    setattr(self, var, filedata[var])
                return True

    def SaveParam(self, filepath):        
        with open(filepath, 'wb+') as f:
            pickle.dump(vars(self), f)
