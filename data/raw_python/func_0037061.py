def getBigIndexFromIndices(self, indices):
        """
        Get the big index from a given set of indices
        @param indices
        @return big index
        @note no checks are performed to ensure that the returned
        indices are valid
        """
        return reduce(operator.add, [self.dimProd[i]*indices[i]
                                     for i in range(self.ndims)], 0)