def getDesc(self, entry):
        """Returns description for stat entry.
        
        @param entry: Entry name.
        @return:      Description for entry.
        
        """
        if len(self._descDict) == 0:
            self.getStats()
        return self._descDict.get(entry)