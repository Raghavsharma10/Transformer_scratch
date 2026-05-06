def populateFromRow(self, dataset):
        """
        Populates the instance variables of this Dataset from the
        specified database row.
        """
        self._description = dataset.description
        self.setAttributesJson(dataset.attributes)