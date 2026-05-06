def shape(self) -> Tuple[int, ...]:
        """Required shape of |NetCDFVariableDeep.array|.

        For the default configuration, the first axis corresponds to the
        number of devices, and the second one to the number of timesteps.
        We show this for the 0-dimensional input sequence |lland_inputs.Nied|:

        >>> from hydpy.core.examples import prepare_io_example_1
        >>> nodes, elements = prepare_io_example_1()
        >>> from hydpy.core.netcdftools import NetCDFVariableDeep
        >>> ncvar = NetCDFVariableDeep('input_nied', isolate=False, timeaxis=1)
        >>> for element in elements:
        ...     ncvar.log(element.model.sequences.inputs.nied, None)
        >>> ncvar.shape
        (3, 4)

        For higher dimensional sequences, each new entry corresponds
        to the maximum number of fields the respective sequences require.
        In the next example, we select the 1-dimensional sequence
        |lland_fluxes.NKor|.  The maximum number 3 (last value of the
        returned |tuple|) is due to the third element defining three
        hydrological response units:

        >>> ncvar = NetCDFVariableDeep('flux_nkor', isolate=False, timeaxis=1)
        >>> for element in elements:
        ...     ncvar.log(element.model.sequences.fluxes.nkor, None)
        >>> ncvar.shape
        (3, 4, 3)

        When using the first axis for time (`timeaxis=0`) the order of the
        first two |tuple| entries turns:

        >>> ncvar = NetCDFVariableDeep('flux_nkor', isolate=False, timeaxis=0)
        >>> for element in elements:
        ...     ncvar.log(element.model.sequences.fluxes.nkor, None)
        >>> ncvar.shape
        (4, 3, 3)
        """
        nmb_place = len(self.sequences)
        nmb_time = len(hydpy.pub.timegrids.init)
        nmb_others = collections.deque()
        for sequence in self.sequences.values():
            nmb_others.append(sequence.shape)
        nmb_others_max = tuple(numpy.max(nmb_others, axis=0))
        return self.sort_timeplaceentries(nmb_time, nmb_place) + nmb_others_max