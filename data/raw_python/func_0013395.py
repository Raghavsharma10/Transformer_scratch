def populateFromRow(self, ontologyRecord):
        """
        Populates this Ontology using values in the specified DB row.
        """
        self._id = ontologyRecord.id
        self._dataUrl = ontologyRecord.dataurl
        self._readFile()