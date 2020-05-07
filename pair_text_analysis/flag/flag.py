import numpy as np

class Flag:
    
    def __init__(self):
        return
        
    def _setFlagPath(self, path=None):
        self._flagPath = path
        
    def getFlags(self, col):
        fileName = self._flagPath
        search_list = [line.rstrip('\n') for line in open(fileName)]
        flagSer = col.apply(lambda x: False if x is np.nan else (True if any(word in search_list for word in str(x).split()) else False))
        return flagSer
