def array(self) -> numpy.ndarray:
        """The series data of all logged |IOSequence| objects contained in
        one single |numpy.ndarray| object.

        The documentation on |NetCDFVariableAgg.shape| explains how
        |NetCDFVariableAgg.array| is structured.  The first example
        confirms that, under default configuration (`timeaxis=1`), the
        first axis corresponds to the location, while the second one
        corresponds to time:

        >>> from hydpy.core.examples import prepare_io_example_1
        >>> nodes, elements = prepare_io_example_1()
        >>> from hydpy.core.netcdftools import NetCDFVariableFlat
        >>> ncvar = NetCDFVariableFlat('input_nied', isolate=False, timeaxis=1)
        >>> for element in elements:
        ...     nied1 = element.model.sequences.inputs.nied
        ...     ncvar.log(nied1, nied1.series)
        >>> ncvar.array
        array([[  0.,   1.,   2.,   3.],
               [  4.,   5.,   6.,   7.],
               [  8.,   9.,  10.,  11.]])

        Due to the flattening of higher dimensional sequences,
        their individual time series (e.g. of different hydrological
        response units) are spread over the rows of the array.
        For the 1-dimensional sequence |lland_fluxes.NKor|, the
        individual time series of the second element are stored
        in row two and three:

        >>> ncvar = NetCDFVariableFlat('flux_nkor', isolate=False, timeaxis=1)
        >>> for element in elements:
        ...     nkor1 = element.model.sequences.fluxes.nkor
        ...     ncvar.log(nkor1, nkor1.series)
        >>> ncvar.array[1:3]
        array([[ 16.,  18.,  20.,  22.],
               [ 17.,  19.,  21.,  23.]])

        When using the first axis as the "timeaxis", the individual time
        series of the second element are stored in column two and three:

        >>> ncvar = NetCDFVariableFlat('flux_nkor', isolate=False, timeaxis=0)
        >>> for element in elements:
        ...     nkor1 = element.model.sequences.fluxes.nkor
        ...     ncvar.log(nkor1, nkor1.series)
        >>> ncvar.array[:, 1:3]
        array([[ 16.,  17.],
               [ 18.,  19.],
               [ 20.,  21.],
               [ 22.,  23.]])
        """
        array = numpy.full(self.shape, fillvalue, dtype=float)
        idx0 = 0
        idxs: List[Any] = [slice(None)]
        for seq, subarray in zip(self.sequences.values(),
                                 self.arrays.values()):
            for prod in self._product(seq.shape):
                subsubarray = subarray[tuple(idxs + list(prod))]
                array[self.get_timeplaceslice(idx0)] = subsubarray
                idx0 += 1
        return array