def dimensions(self) -> Tuple[str, ...]:
        """The dimension names of the NetCDF variable.

        Usually, the string defined by property |IOSequence.descr_sequence|
        prefixes the first dimension name related to the location, which
        allows storing different sequences types in one NetCDF file:

        >>> from hydpy.core.examples import prepare_io_example_1
        >>> nodes, elements = prepare_io_example_1()
        >>> from hydpy.core.netcdftools import NetCDFVariableAgg
        >>> ncvar = NetCDFVariableAgg('flux_nkor', isolate=False, timeaxis=1)
        >>> ncvar.log(elements.element1.model.sequences.fluxes.nkor, None)
        >>> ncvar.dimensions
        ('flux_nkor_stations', 'time')

        But when isolating variables into separate NetCDF files, the
        variable specific suffix is omitted:

        >>> ncvar = NetCDFVariableAgg('flux_nkor', isolate=True, timeaxis=1)
        >>> ncvar.log(elements.element1.model.sequences.fluxes.nkor, None)
        >>> ncvar.dimensions
        ('stations', 'time')

        When using the first axis as the "timeaxis", the order of the
        dimension names turns:

        >>> ncvar = NetCDFVariableAgg('flux_nkor', isolate=True, timeaxis=0)
        >>> ncvar.log(elements.element1.model.sequences.fluxes.nkor, None)
        >>> ncvar.dimensions
        ('time', 'stations')
        """
        self: NetCDFVariableBase
        return self.sort_timeplaceentries(
            dimmapping['nmb_timepoints'],
            '%s%s' % (self.prefix, dimmapping['nmb_subdevices']))