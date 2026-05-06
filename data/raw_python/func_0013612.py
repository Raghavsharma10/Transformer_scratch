def populateFromRow(self, referenceSetRecord):
        """
        Populates this reference set from the values in the specified DB
        row.
        """
        self._dataUrl = referenceSetRecord.dataurl
        self._description = referenceSetRecord.description
        self._assemblyId = referenceSetRecord.assemblyid
        self._isDerived = bool(referenceSetRecord.isderived)
        self._md5checksum = referenceSetRecord.md5checksum
        species = referenceSetRecord.species
        if species is not None and species != 'null':
            self.setSpeciesFromJson(species)
        self._sourceAccessions = json.loads(
            referenceSetRecord.sourceaccessions)
        self._sourceUri = referenceSetRecord.sourceuri