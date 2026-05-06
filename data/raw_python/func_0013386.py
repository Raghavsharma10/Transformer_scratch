def populateFromFile(self, dataUrl):
        """
        Populates the instance variables of this FeatureSet from the specified
        data URL.
        Initialize dataset, using the passed dict of sources
        [{source,format}] see rdflib.parse() for more
        If path is set, this backend will load itself
        """
        self._dbFilePath = dataUrl
        # initialize graph
        self._rdfGraph = rdflib.ConjunctiveGraph()
        # save the path
        self._dataUrl = dataUrl
        self._scanDataFiles(self._dataUrl, ['*.ttl'])

        # extract version
        cgdTTL = rdflib.URIRef("http://data.monarchinitiative.org/ttl/cgd.ttl")
        versionInfo = rdflib.URIRef(
            u'http://www.w3.org/2002/07/owl#versionInfo')
        self._version = None
        for _, _, obj in self._rdfGraph.triples((cgdTTL, versionInfo, None)):
            self._version = obj.toPython()

        # setup location cache
        self._initializeLocationCache()