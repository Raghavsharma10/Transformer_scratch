def getPysamVariants(self, referenceName, startPosition, endPosition):
        """
        Returns an iterator over the pysam VCF records corresponding to the
        specified query.
        """
        if referenceName in self._chromFileMap:
            varFileName = self._chromFileMap[referenceName]
            referenceName, startPosition, endPosition = \
                self.sanitizeVariantFileFetch(
                    referenceName, startPosition, endPosition)
            cursor = self.getFileHandle(varFileName).fetch(
                referenceName, startPosition, endPosition)
            for record in cursor:
                yield record