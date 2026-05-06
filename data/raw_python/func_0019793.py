def restoreState(self):
        """Utility method to restore plugin state from persistent storage to 
        permit access to previous plugin state.
        
        @return: Object that stores plugin state.
        
        """
        if os.path.exists(self._stateFile):
            try:
                fp = open(self._stateFile,  'r')
                stateObj = pickle.load(fp)
            except:
                raise IOError("Failure in reading plugin state from file: %s" 
                              % self._stateFile)
            return stateObj
        return None