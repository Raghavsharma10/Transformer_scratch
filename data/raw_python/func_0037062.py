def areIndicesValid(self, inds):
        """
        Test if indices are valid
        @param inds index set
        @return True if valid, False otherwise
        """
        return reduce(operator.and_, [0 <= inds[d] < self.dims[d]
                                      for d in range(self.ndims)], True)