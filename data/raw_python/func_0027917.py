def _fixIndex(self, index, truncate=False):
        """
        @param truncate: If true, negative indices which go past the
                         beginning of the list will be evaluated as zero.
                         For example::

                         >>> L = List([1,2,3,4,5])
                         >>> len(L)
                         5
                         >>> L._fixIndex(-9, truncate=True)
                         0
        """
        assert not isinstance(index, slice), 'slices are not supported (yet)'
        if index < 0:
            index += self.length
        if index < 0:
            if not truncate:
                raise IndexError('stored List index out of range')
            else:
                index = 0
        return index