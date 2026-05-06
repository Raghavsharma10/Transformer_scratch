def getVals(self):
        """Returns value list for Munin Graph
        
        @return: List of name-value pairs.
        
        """
        return [(name, self._fieldValDict.get(name)) 
                for name in self._fieldNameList]