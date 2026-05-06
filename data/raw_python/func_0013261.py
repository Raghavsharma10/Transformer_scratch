def populateFromRow(self, quantificationSetRecord):
        """
        Populates the instance variables of this RnaQuantificationSet from the
        specified DB row.
        """
        self._dbFilePath = quantificationSetRecord.dataurl
        self.setAttributesJson(quantificationSetRecord.attributes)
        self._db = SqliteRnaBackend(self._dbFilePath)
        self.addRnaQuants()