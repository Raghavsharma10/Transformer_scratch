def groupby(self, io_select):
        """
        SLICE THIS MATRIX INTO ONES WITH LESS DIMENSIONALITY
        io_select - 1 IF GROUPING BY THIS DIMENSION, 0 IF FLATTENING
        return -
        """

        # offsets WILL SERVE TO MASK DIMS WE ARE NOT GROUPING BY, AND SERVE AS RELATIVE INDEX FOR EACH COORDINATE
        offsets = []
        new_dim = []
        acc = 1
        for i, d in reversed(enumerate(self.dims)):
            if not io_select[i]:
                new_dim.insert(0, d)
            offsets.insert(0, acc * io_select[i])
            acc *= d

        if not new_dim:
            # WHEN groupby ALL DIMENSIONS, ONLY THE VALUES REMAIN
            # RETURN AN ITERATOR OF PAIRS (c, v), WHERE
            # c - COORDINATES INTO THE CUBE
            # v - VALUE AT GIVEN COORDINATES
            return ((c, self[c]) for c in self._all_combos())
        else:
            output = [[None, Matrix(dims=new_dim)] for i in range(acc)]
            _groupby(self.cube, 0, offsets, 0, output, tuple(), [])

        return output