def log(self, sequence, infoarray) -> None:
        """Log the given |IOSequence| object either for reading or writing
        data.

        The optional `array` argument allows for passing alternative data
        in an |InfoArray| object replacing the series of the |IOSequence|
        object, which is useful for writing modified (e.g. spatially
        averaged) time series.

        Logged time series data is available via attribute access:

        >>> from hydpy.core.netcdftools import NetCDFVariableBase
        >>> from hydpy import make_abc_testable
        >>> NCVar = make_abc_testable(NetCDFVariableBase)
        >>> ncvar = NCVar('flux_nkor', isolate=True, timeaxis=1)
        >>> from hydpy.core.examples import prepare_io_example_1
        >>> nodes, elements = prepare_io_example_1()
        >>> nkor = elements.element1.model.sequences.fluxes.nkor
        >>> ncvar.log(nkor, nkor.series)
        >>> 'element1' in dir(ncvar)
        True
        >>> ncvar.element1.sequence is nkor
        True
        >>> 'element2' in dir(ncvar)
        False
        >>> ncvar.element2
        Traceback (most recent call last):
        ...
        AttributeError: The NetCDFVariable object `flux_nkor` does \
neither handle time series data under the (sub)device name `element2` \
nor does it define a member named `element2`.
        """
        descr_device = sequence.descr_device
        self.sequences[descr_device] = sequence
        self.arrays[descr_device] = infoarray