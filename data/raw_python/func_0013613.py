def populateFromRow(self, referenceRecord):
        """
        Populates this reference from the values in the specified DB row.
        """
        self._length = referenceRecord.length
        self._isDerived = bool(referenceRecord.isderived)
        self._md5checksum = referenceRecord.md5checksum
        species = referenceRecord.species
        if species is not None and species != 'null':
            self.setSpeciesFromJson(species)
        self._sourceAccessions = json.loads(referenceRecord.sourceaccessions)
        self._sourceDivergence = referenceRecord.sourcedivergence
        self._sourceUri = referenceRecord.sourceuri