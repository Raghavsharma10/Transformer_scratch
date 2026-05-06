def addReadGroupSet(self):
        """
        Adds a new ReadGroupSet into this repo.
        """
        self._openRepo()
        dataset = self._repo.getDatasetByName(self._args.datasetName)
        dataUrl = self._args.dataFile
        indexFile = self._args.indexFile
        parsed = urlparse.urlparse(dataUrl)
        # TODO, add https support and others when they have been
        # tested.
        if parsed.scheme in ['http', 'ftp']:
            if indexFile is None:
                raise exceptions.MissingIndexException(dataUrl)
        else:
            if indexFile is None:
                indexFile = dataUrl + ".bai"
            dataUrl = self._getFilePath(self._args.dataFile,
                                        self._args.relativePath)
            indexFile = self._getFilePath(indexFile, self._args.relativePath)
        name = self._args.name
        if self._args.name is None:
            name = getNameFromPath(dataUrl)
        readGroupSet = reads.HtslibReadGroupSet(dataset, name)
        readGroupSet.populateFromFile(dataUrl, indexFile)
        referenceSetName = self._args.referenceSetName
        if referenceSetName is None:
            # Try to find a reference set name from the BAM header.
            referenceSetName = readGroupSet.getBamHeaderReferenceSetName()
        referenceSet = self._repo.getReferenceSetByName(referenceSetName)
        readGroupSet.setReferenceSet(referenceSet)
        readGroupSet.setAttributes(json.loads(self._args.attributes))
        self._updateRepo(self._repo.insertReadGroupSet, readGroupSet)