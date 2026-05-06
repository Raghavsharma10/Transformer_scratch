def _check_index(self, index):
        """Verify that the given index is consistent with the degree of the node.
        """
        if self.degree is None:
            raise UnknownDegreeError(
                'Cannot access child DataNode on a parent with degree of None. '\
                'Set the degree on the parent first.')
        if index < 0 or index >= self.degree:
            raise IndexOutOfRangeError(
                'Out of range index %s. DataNode parent has degree %s, so index '\
                'should be in the range 0 to %s' % (
                    index, self.degree, self.degree-1))