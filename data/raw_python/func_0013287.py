def populateFromFile(self, dataUrls, indexFiles):
        """
        Populates this variant set using the specified lists of data
        files and indexes. These must be in the same order, such that
        the jth index file corresponds to the jth data file.
        """
        assert len(dataUrls) == len(indexFiles)
        for dataUrl, indexFile in zip(dataUrls, indexFiles):
            varFile = pysam.VariantFile(dataUrl, index_filename=indexFile)
            try:
                self._populateFromVariantFile(varFile, dataUrl, indexFile)
            finally:
                varFile.close()