def initialise(self):
        """
        Initialise this data repository, creating any necessary directories
        and file paths.
        """
        self._checkWriteMode()
        self._createSystemTable()
        self._createNetworkTables()
        self._createOntologyTable()
        self._createReferenceSetTable()
        self._createReferenceTable()
        self._createDatasetTable()
        self._createReadGroupSetTable()
        self._createReadGroupTable()
        self._createCallSetTable()
        self._createVariantSetTable()
        self._createVariantAnnotationSetTable()
        self._createFeatureSetTable()
        self._createContinuousSetTable()
        self._createBiosampleTable()
        self._createIndividualTable()
        self._createPhenotypeAssociationSetTable()
        self._createRnaQuantificationSetTable()