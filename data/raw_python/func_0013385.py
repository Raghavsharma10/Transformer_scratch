def populateFromRow(self, featureSetRecord):
        """
        Populates the instance variables of this FeatureSet from the specified
        DB row.
        """
        self._dbFilePath = featureSetRecord.dataurl
        self.setAttributesJson(featureSetRecord.attributes)
        self.populateFromFile(self._dbFilePath)