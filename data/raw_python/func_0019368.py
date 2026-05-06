def sort_timeplaceentries(self, timeentry, placeentry) -> Tuple[Any, Any]:
        """Return a |tuple| containing the given `timeentry` and `placeentry`
        sorted in agreement with the currently selected `timeaxis`.

        >>> from hydpy.core.netcdftools import NetCDFVariableBase
        >>> from hydpy import make_abc_testable
        >>> NCVar = make_abc_testable(NetCDFVariableBase)
        >>> ncvar = NCVar('flux_nkor', isolate=True, timeaxis=1)
        >>> ncvar.sort_timeplaceentries('time', 'place')
        ('place', 'time')
        >>> ncvar = NetCDFVariableDeep('test', isolate=False, timeaxis=0)
        >>> ncvar.sort_timeplaceentries('time', 'place')
        ('time', 'place')
        """
        if self._timeaxis:
            return placeentry, timeentry
        return timeentry, placeentry