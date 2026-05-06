def populateFromRow(self, continuousSetRecord):
        """
        Populates the instance variables of this ContinuousSet from the
        specified DB row.
        """
        self._filePath = continuousSetRecord.dataurl
        self.setAttributesJson(continuousSetRecord.attributes)