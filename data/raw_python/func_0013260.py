def populateFromFile(self, dataUrl):
        """
        Populates the instance variables of this RnaQuantificationSet from the
        specified data URL.
        """
        self._dbFilePath = dataUrl
        self._db = SqliteRnaBackend(self._dbFilePath)
        self.addRnaQuants()