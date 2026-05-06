def load(self):
        """
        Loads this data repository into memory.
        """
        self._readSystemTable()
        self._readOntologyTable()
        self._readReferenceSetTable()
        self._readReferenceTable()
        self._readDatasetTable()
        self._readReadGroupSetTable()
        self._readReadGroupTable()
        self._readVariantSetTable()
        self._readCallSetTable()
        self._readVariantAnnotationSetTable()
        self._readFeatureSetTable()
        self._readContinuousSetTable()
        self._readBiosampleTable()
        self._readIndividualTable()
        self._readPhenotypeAssociationSetTable()
        self._readRnaQuantificationSetTable()