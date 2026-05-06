def subdevicenames(self) -> Tuple[str, ...]:
        """A |tuple| containing the (sub)device names.

        Property |NetCDFVariableFlat.subdevicenames| clarifies which
        row of |NetCDFVariableAgg.array| contains which time series.
        For 0-dimensional series like |lland_inputs.Nied|, the plain
        device names are returned

        >>> from hydpy.core.examples import prepare_io_example_1
        >>> nodes, elements = prepare_io_example_1()
        >>> from hydpy.core.netcdftools import NetCDFVariableFlat
        >>> ncvar = NetCDFVariableFlat('input_nied', isolate=False, timeaxis=1)
        >>> for element in elements:
        ...     nied1 = element.model.sequences.inputs.nied
        ...     ncvar.log(nied1, nied1.series)
        >>> ncvar.subdevicenames
        ('element1', 'element2', 'element3')

        For higher dimensional sequences like |lland_fluxes.NKor|, an
        additional suffix defines the index of the respective subdevice.
        For example contains the third row of |NetCDFVariableAgg.array|
        the time series of the first hydrological response unit of the
        second element:

        >>> ncvar = NetCDFVariableFlat('flux_nkor', isolate=False, timeaxis=1)
        >>> for element in elements:
        ...     nkor1 = element.model.sequences.fluxes.nkor
        ...     ncvar.log(nkor1, nkor1.series)
        >>> ncvar.subdevicenames[1:3]
        ('element2_0', 'element2_1')
        """
        stats: List[str] = collections.deque()
        for devicename, seq in self.sequences.items():
            if seq.NDIM:
                temp = devicename + '_'
                for prod in self._product(seq.shape):
                    stats.append(temp + '_'.join(str(idx) for idx in prod))
            else:
                stats.append(devicename)
        return tuple(stats)