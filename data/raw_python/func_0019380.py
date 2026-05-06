def shape(self) -> Tuple[int, int]:
        """Required shape of |NetCDFVariableFlat.array|.

        For 0-dimensional sequences like |lland_inputs.Nied| and for the
        default configuration (`timeaxis=1`), the first axis corresponds
        to the number of devices, and the second one two the number of
        timesteps:

        >>> from hydpy.core.examples import prepare_io_example_1
        >>> nodes, elements = prepare_io_example_1()
        >>> from hydpy.core.netcdftools import NetCDFVariableFlat
        >>> ncvar = NetCDFVariableFlat('input_nied', isolate=False, timeaxis=1)
        >>> for element in elements:
        ...     ncvar.log(element.model.sequences.inputs.nied, None)
        >>> ncvar.shape
        (3, 4)

        For higher dimensional sequences, the first axis corresponds
        to "subdevices", e.g. hydrological response units within
        different elements.  The 1-dimensional sequence |lland_fluxes.NKor|
        is logged for three elements with one, two, and three response
        units respectively, making up a sum of six subdevices:

        >>> ncvar = NetCDFVariableFlat('flux_nkor', isolate=False, timeaxis=1)
        >>> for element in elements:
        ...     ncvar.log(element.model.sequences.fluxes.nkor, None)
        >>> ncvar.shape
        (6, 4)

        When using the first axis as the "timeaxis", the order of |tuple|
        entries turns:

        >>> ncvar = NetCDFVariableFlat('flux_nkor', isolate=False, timeaxis=0)
        >>> for element in elements:
        ...     ncvar.log(element.model.sequences.fluxes.nkor, None)
        >>> ncvar.shape
        (4, 6)
        """
        return self.sort_timeplaceentries(
            len(hydpy.pub.timegrids.init),
            sum(len(seq) for seq in self.sequences.values()))