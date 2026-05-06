def get_slices(self, idx, shape) -> Tuple[IntOrSlice, ...]:
        """Return a |tuple| of one |int| and some |slice| objects to
        accesses all values of a certain device within
        |NetCDFVariableDeep.array|.

        >>> from hydpy.core.netcdftools import NetCDFVariableDeep
        >>> ncvar = NetCDFVariableDeep('test', isolate=False, timeaxis=1)
        >>> ncvar.get_slices(2, [3])
        (2, slice(None, None, None), slice(0, 3, None))
        >>> ncvar.get_slices(4, (1, 2))
        (4, slice(None, None, None), slice(0, 1, None), slice(0, 2, None))
        >>> ncvar = NetCDFVariableDeep('test', isolate=False, timeaxis=0)
        >>> ncvar.get_slices(4, (1, 2))
        (slice(None, None, None), 4, slice(0, 1, None), slice(0, 2, None))
        """
        slices = list(self.get_timeplaceslice(idx))
        for length in shape:
            slices.append(slice(0, length))
        return tuple(slices)