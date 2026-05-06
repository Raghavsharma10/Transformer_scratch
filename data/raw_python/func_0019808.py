def setVal(self, name, val):
        """Set value for field in graph.
        
        @param name   : Graph Name
        @param value  : Value for field. 
        
        """
        if self._autoFixNames:
            name = self._fixName(name)
        self._fieldValDict[name] = val