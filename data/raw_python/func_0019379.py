def array(self) -> numpy.ndarray:
        """The aggregated data of all logged |IOSequence| objects contained
        in one single |numpy.ndarray| object.

        The documentation on |NetCDFVariableAgg.shape| explains how
        |NetCDFVariableAgg.array| is structured.  This first example
        confirms that, under default configuration (`timeaxis=1`),
        the first axis corresponds to the location, while the second
        one corresponds to time:

        >>> from hydpy.core.examples import prepare_io_example_1
        >>> nodes, elements = prepare_io_example_1()
        >>> from hydpy.core.netcdftools import NetCDFVariableAgg
        >>> ncvar = NetCDFVariableAgg('flux_nkor', isolate=False, timeaxis=1)
        >>> for element in elements:
        ...     nkor1 = element.model.sequences.fluxes.nkor
        ...     ncvar.log(nkor1, nkor1.average_series())
        >>> ncvar.array
        array([[ 12. ,  13. ,  14. ,  15. ],
               [ 16.5,  18.5,  20.5,  22.5],
               [ 25. ,  28. ,  31. ,  34. ]])

        When using the first axis as the "timeaxis", the resulting
        |NetCDFVariableAgg.array| is the transposed:

        >>> ncvar = NetCDFVariableAgg('flux_nkor', isolate=False, timeaxis=0)
        >>> for element in elements:
        ...     nkor1 = element.model.sequences.fluxes.nkor
        ...     ncvar.log(nkor1, nkor1.average_series())
        >>> ncvar.array
        array([[ 12. ,  16.5,  25. ],
               [ 13. ,  18.5,  28. ],
               [ 14. ,  20.5,  31. ],
               [ 15. ,  22.5,  34. ]])
        """
        array = numpy.full(self.shape, fillvalue, dtype=float)
        for idx, subarray in enumerate(self.arrays.values()):
            array[self.get_timeplaceslice(idx)] = subarray
        return array