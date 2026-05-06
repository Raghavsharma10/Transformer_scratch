def getIndicesFromBigIndex(self, bigIndex):
        """
        Get index set from given big index
        @param bigIndex
        @return index set
        @note no checks are performed to ensure that the returned
        big index is valid
        """
        indices = numpy.array([0 for i in range(self.ndims)])
        for i in range(self.ndims):
            indices[i] = bigIndex // self.dimProd[i] % self.dims[i]
        return indices