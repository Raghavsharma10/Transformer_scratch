def checkConsistency(self):
        """
        Perform consistency check on the variant set
        """
        for referenceName, (dataUrl, indexFile) in self._chromFileMap.items():
            varFile = pysam.VariantFile(dataUrl, index_filename=indexFile)
            try:
                for chrom in varFile.index:
                    chrom, _, _ = self.sanitizeVariantFileFetch(chrom)
                    if not isEmptyIter(varFile.fetch(chrom)):
                        self._checkMetadata(varFile)
                        self._checkCallSetIds(varFile)
            finally:
                varFile.close()