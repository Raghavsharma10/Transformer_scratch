def shape(self) -> Tuple[int, int]:
        """Required shape of |NetCDFVariableAgg.array|.

        For the default configuration, the first axis corresponds to the
        number of devices, and the second one to the number of timesteps.
        We show this for the 1-dimensional input sequence |lland_fluxes.NKor|:

        >>> from hydpy.core.examples import prepare_io_example_1
        >>> nodes, elements = prepare_io_example_1()
        >>> from hydpy.core.netcdftools import NetCDFVariableAgg
        >>> ncvar = NetCDFVariableAgg('flux_nkor', isolate=False, timeaxis=1)
        >>> for element in elements:
        ...     ncvar.log(element.model.sequences.fluxes.nkor, None)
        >>> ncvar.shape
        (3, 4)

        When using the first axis as the "timeaxis", the order of |tuple|
        entries turns:

        >>> ncvar = NetCDFVariableAgg('flux_nkor', isolate=False, timeaxis=0)
        >>> for element in elements:
        ...     ncvar.log(element.model.sequences.fluxes.nkor, None)
        >>> ncvar.shape
        (4, 3)
        """
        return self.sort_timeplaceentries(
            len(hydpy.pub.timegrids.init), len(self.sequences))