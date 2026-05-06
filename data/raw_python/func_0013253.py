def populateFromFile(self, dataUrl):
        """
        Populates the instance variables of this FeatureSet from the specified
        data URL.
        """
        self._dbFilePath = dataUrl
        self._db = Gff3DbBackend(self._dbFilePath)