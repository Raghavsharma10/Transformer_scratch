def get_timeplaceslice(self, placeindex) -> \
            Union[Tuple[slice, int], Tuple[int, slice]]:
        """Return a |tuple| for indexing a complete time series of a certain
        location available in |NetCDFVariableBase.array|.

        >>> from hydpy.core.netcdftools import NetCDFVariableBase
        >>> from hydpy import make_abc_testable
        >>> NCVar = make_abc_testable(NetCDFVariableBase)
        >>> ncvar = NCVar('flux_nkor', isolate=True, timeaxis=1)
        >>> ncvar.get_timeplaceslice(2)
        (2, slice(None, None, None))
        >>> ncvar = NetCDFVariableDeep('test', isolate=False, timeaxis=0)
        >>> ncvar.get_timeplaceslice(2)
        (slice(None, None, None), 2)
        """
        return self.sort_timeplaceentries(slice(None), int(placeindex))