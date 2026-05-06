def _populateFromVariantFile(self, varFile, dataUrl, indexFile):
        """
        Populates the instance variables of this VariantSet from the specified
        pysam VariantFile object.
        """
        if varFile.index is None:
            raise exceptions.NotIndexedException(dataUrl)
        for chrom in varFile.index:
            # Unlike Tabix indices, CSI indices include all contigs defined
            # in the BCF header.  Thus we must test each one to see if
            # records exist or else they are likely to trigger spurious
            # overlapping errors.
            chrom, _, _ = self.sanitizeVariantFileFetch(chrom)
            if not isEmptyIter(varFile.fetch(chrom)):
                if chrom in self._chromFileMap:
                    raise exceptions.OverlappingVcfException(dataUrl, chrom)
            self._chromFileMap[chrom] = dataUrl, indexFile
        self._updateMetadata(varFile)
        self._updateCallSetIds(varFile)
        self._updateVariantAnnotationSets(varFile, dataUrl)