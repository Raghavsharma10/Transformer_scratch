def populateFromFile(self, dataUrl, indexFile=None):
        """
        Populates the instance variables of this ReadGroupSet from the
        specified dataUrl and indexFile. If indexFile is not specified
        guess usual form.
        """
        self._dataUrl = dataUrl
        self._indexFile = indexFile
        if indexFile is None:
            self._indexFile = dataUrl + ".bai"
        samFile = self.getFileHandle(self._dataUrl)
        self._setHeaderFields(samFile)
        if 'RG' not in samFile.header or len(samFile.header['RG']) == 0:
            readGroup = HtslibReadGroup(self, self.defaultReadGroupName)
            self.addReadGroup(readGroup)
        else:
            for readGroupHeader in samFile.header['RG']:
                readGroup = HtslibReadGroup(self, readGroupHeader['ID'])
                readGroup.populateFromHeader(readGroupHeader)
                self.addReadGroup(readGroup)
        self._bamHeaderReferenceSetName = None
        for referenceInfo in samFile.header['SQ']:
            if 'AS' not in referenceInfo:
                infoDict = parseMalformedBamHeader(referenceInfo)
            else:
                infoDict = referenceInfo
            name = infoDict.get('AS', references.DEFAULT_REFERENCESET_NAME)
            if self._bamHeaderReferenceSetName is None:
                self._bamHeaderReferenceSetName = name
            elif self._bamHeaderReferenceSetName != name:
                raise exceptions.MultipleReferenceSetsInReadGroupSet(
                    self._dataUrl, name, self._bamFileReferenceName)
        self._numAlignedReads = samFile.mapped
        self._numUnalignedReads = samFile.unmapped