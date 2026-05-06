def array(self) -> numpy.ndarray:
        """The series data of all logged |IOSequence| objects contained
        in one single |numpy.ndarray|.

        The documentation on |NetCDFVariableDeep.shape| explains how
        |NetCDFVariableDeep.array| is structured.  The first example
        confirms that, for the default configuration, the first axis
        definces the location, while the second one defines time:

        >>> from hydpy.core.examples import prepare_io_example_1
        >>> nodes, elements = prepare_io_example_1()
        >>> from hydpy.core.netcdftools import NetCDFVariableDeep
        >>> ncvar = NetCDFVariableDeep('input_nied', isolate=False, timeaxis=1)
        >>> for element in elements:
        ...     nied1 = element.model.sequences.inputs.nied
        ...     ncvar.log(nied1, nied1.series)
        >>> ncvar.array
        array([[  0.,   1.,   2.,   3.],
               [  4.,   5.,   6.,   7.],
               [  8.,   9.,  10.,  11.]])

        For higher dimensional sequences, |NetCDFVariableDeep.array|
        can contain missing values.  Such missing values show up for
        some fiels of the second example element, which defines only
        two hydrological response units instead of three:

        >>> ncvar = NetCDFVariableDeep('flux_nkor', isolate=False, timeaxis=1)
        >>> for element in elements:
        ...     nkor1 = element.model.sequences.fluxes.nkor
        ...     ncvar.log(nkor1, nkor1.series)
        >>> ncvar.array[1]
        array([[ 16.,  17.,  nan],
               [ 18.,  19.,  nan],
               [ 20.,  21.,  nan],
               [ 22.,  23.,  nan]])

        When using the first axis for time (`timeaxis=0`) the same data
        can be accessed with slightly different indexing:

        >>> ncvar = NetCDFVariableDeep('flux_nkor', isolate=False, timeaxis=0)
        >>> for element in elements:
        ...     nkor1 = element.model.sequences.fluxes.nkor
        ...     ncvar.log(nkor1, nkor1.series)
        >>> ncvar.array[:, 1]
        array([[ 16.,  17.,  nan],
               [ 18.,  19.,  nan],
               [ 20.,  21.,  nan],
               [ 22.,  23.,  nan]])
        """
        array = numpy.full(self.shape, fillvalue, dtype=float)
        for idx, (descr, subarray) in enumerate(self.arrays.items()):
            sequence = self.sequences[descr]
            array[self.get_slices(idx, sequence.shape)] = subarray
        return array