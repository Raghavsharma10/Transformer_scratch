def dimensions(self) -> Tuple[str, ...]:
        """The dimension names of the NetCDF variable.

        Usually, the string defined by property |IOSequence.descr_sequence|
        prefixes all dimension names except the second one related to time,
        which allows storing different sequences in one NetCDF file:

        >>> from hydpy.core.examples import prepare_io_example_1
        >>> nodes, elements = prepare_io_example_1()
        >>> from hydpy.core.netcdftools import NetCDFVariableDeep
        >>> ncvar = NetCDFVariableDeep('flux_nkor', isolate=False, timeaxis=1)
        >>> ncvar.log(elements.element1.model.sequences.fluxes.nkor, None)
        >>> ncvar.dimensions
        ('flux_nkor_stations', 'time', 'flux_nkor_axis3')

        However, when isolating variables into separate NetCDF files, the
        sequence-specific suffix is omitted:

        >>> ncvar = NetCDFVariableDeep('flux_nkor', isolate=True, timeaxis=1)
        >>> ncvar.log(elements.element1.model.sequences.fluxes.nkor, None)
        >>> ncvar.dimensions
        ('stations', 'time', 'axis3')

        When using the first axis as the "timeaxis", the order of the
        first two dimension names turns:

        >>> ncvar = NetCDFVariableDeep('flux_nkor', isolate=True, timeaxis=0)
        >>> ncvar.log(elements.element1.model.sequences.fluxes.nkor, None)
        >>> ncvar.dimensions
        ('time', 'stations', 'axis3')
        """
        nmb_timepoints = dimmapping['nmb_timepoints']
        nmb_subdevices = '%s%s' % (self.prefix, dimmapping['nmb_subdevices'])
        dimensions = list(self.sort_timeplaceentries(
            nmb_timepoints, nmb_subdevices))
        for idx in range(list(self.sequences.values())[0].NDIM):
            dimensions.append('%saxis%d' % (self.prefix, idx + 3))
        return tuple(dimensions)