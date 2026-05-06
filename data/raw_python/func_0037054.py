def getNeighborProc(self, proc, offset, periodic=None):
        """
        Get the neighbor to a processor
        @param proc the reference processor rank
        @param offset displacement, e.g. (1, 0) for north, (0, -1) for west,...
        @param periodic boolean list of True/False values, True if axis is
                        periodic, False otherwise
        @note will return None if there is no neighbor
        """

        if self.mit is None:
            # no decomp, just exit
            return None

        inds = [self.mit.getIndicesFromBigIndex(proc)[d] + offset[d]
                for d in range(self.ndims)]

        if periodic is not None and self.decomp is not None:
            # apply modulo operation on periodic axes
            for d in range(self.ndims):
                if periodic[d]:
                    inds[d] = inds[d] % self.decomp[d]

        if self.mit.areIndicesValid(inds):
            return self.mit.getBigIndexFromIndices(inds)
        else:
            return None