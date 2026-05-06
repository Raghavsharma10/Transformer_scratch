def saveState(self,  stateObj):
        """Utility methos to save plugin state stored in stateObj to persistent 
        storage to permit access to previous state in subsequent plugin runs.
        
        Any object that can be pickled and unpickled can be used to store the 
        plugin state.
        
        @param stateObj: Object that stores plugin state.
        
        """
        try:
            fp = open(self._stateFile,  'w')
            pickle.dump(stateObj, fp)
        except:
            raise IOError("Failure in storing plugin state in file: %s" 
                          % self._stateFile)
        return True