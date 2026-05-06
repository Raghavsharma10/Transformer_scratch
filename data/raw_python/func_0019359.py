def log(self, sequence, infoarray) -> None:
        """Pass the given |IoSequence| to a suitable instance of
        a |NetCDFVariableBase| subclass.

        When writing data, the second argument should be an |InfoArray|.
        When reading data, this argument is ignored. Simply pass |None|.

        (1) We prepare some devices handling some sequences by applying
        function |prepare_io_example_1|.  We limit our attention to the
        returned elements, which handle the more diverse sequences:

        >>> from hydpy.core.examples import prepare_io_example_1
        >>> nodes, (element1, element2, element3) = prepare_io_example_1()

        (2) We define some shortcuts for the sequences used in the
        following examples:

        >>> nied1 = element1.model.sequences.inputs.nied
        >>> nied2 = element2.model.sequences.inputs.nied
        >>> nkor2 = element2.model.sequences.fluxes.nkor
        >>> nkor3 = element3.model.sequences.fluxes.nkor

        (3) We define a function that logs these example sequences
        to a given |NetCDFFile| object and prints some information
        about the resulting object structure.  Note that sequence
        `nkor2` is logged twice, the first time with its original
        time series data, the second time with averaged values:

        >>> from hydpy import classname
        >>> def test(ncfile):
        ...     ncfile.log(nied1, nied1.series)
        ...     ncfile.log(nied2, nied2.series)
        ...     ncfile.log(nkor2, nkor2.series)
        ...     ncfile.log(nkor2, nkor2.average_series())
        ...     ncfile.log(nkor3, nkor3.average_series())
        ...     for name, variable in ncfile.variables.items():
        ...         print(name, classname(variable), variable.subdevicenames)

        (4) We prepare a |NetCDFFile| object with both options
        `flatten` and `isolate` being disabled:

        >>> from hydpy.core.netcdftools import NetCDFFile
        >>> ncfile = NetCDFFile(
        ...     'model', flatten=False, isolate=False, timeaxis=1, dirpath='')

        (5) We log all test sequences results in two |NetCDFVariableDeep|
        and one |NetCDFVariableAgg| objects.  To keep both NetCDF variables
        related to |lland_fluxes.NKor| distinguishable, the name
        `flux_nkor_mean` includes information about the kind of aggregation
        performed:

        >>> test(ncfile)
        input_nied NetCDFVariableDeep ('element1', 'element2')
        flux_nkor NetCDFVariableDeep ('element2',)
        flux_nkor_mean NetCDFVariableAgg ('element2', 'element3')

        (6) We confirm that the |NetCDFVariableBase| objects received
        the required information:

        >>> ncfile.flux_nkor.element2.sequence.descr_device
        'element2'
        >>> ncfile.flux_nkor.element2.array
        InfoArray([[ 16.,  17.],
                   [ 18.,  19.],
                   [ 20.,  21.],
                   [ 22.,  23.]])
        >>> ncfile.flux_nkor_mean.element2.sequence.descr_device
        'element2'
        >>> ncfile.flux_nkor_mean.element2.array
        InfoArray([ 16.5,  18.5,  20.5,  22.5])

        (7) We again prepare a |NetCDFFile| object, but now with both
        options `flatten` and `isolate` being enabled.  To log test
        sequences with their original time series data does now trigger
        the initialisation of class |NetCDFVariableFlat|.  When passing
        aggregated data, nothing changes:

        >>> ncfile = NetCDFFile(
        ...     'model', flatten=True, isolate=True, timeaxis=1, dirpath='')
        >>> test(ncfile)
        input_nied NetCDFVariableFlat ('element1', 'element2')
        flux_nkor NetCDFVariableFlat ('element2_0', 'element2_1')
        flux_nkor_mean NetCDFVariableAgg ('element2', 'element3')
        >>> ncfile.flux_nkor.element2.sequence.descr_device
        'element2'
        >>> ncfile.flux_nkor.element2.array
        InfoArray([[ 16.,  17.],
                   [ 18.,  19.],
                   [ 20.,  21.],
                   [ 22.,  23.]])
        >>> ncfile.flux_nkor_mean.element2.sequence.descr_device
        'element2'
        >>> ncfile.flux_nkor_mean.element2.array
        InfoArray([ 16.5,  18.5,  20.5,  22.5])

        (8) We technically confirm that the `isolate` argument is passed
        to the constructor of subclasses of |NetCDFVariableBase| correctly:

        >>> from unittest.mock import patch
        >>> with patch('hydpy.core.netcdftools.NetCDFVariableFlat') as mock:
        ...     ncfile = NetCDFFile(
        ...         'model', flatten=True, isolate=False, timeaxis=0,
        ...         dirpath='')
        ...     ncfile.log(nied1, nied1.series)
        ...     mock.assert_called_once_with(
        ...         name='input_nied', timeaxis=0, isolate=False)
        """
        aggregated = ((infoarray is not None) and
                      (infoarray.info['type'] != 'unmodified'))
        descr = sequence.descr_sequence
        if aggregated:
            descr = '_'.join([descr, infoarray.info['type']])
        if descr in self.variables:
            var_ = self.variables[descr]
        else:
            if aggregated:
                cls = NetCDFVariableAgg
            elif self._flatten:
                cls = NetCDFVariableFlat
            else:
                cls = NetCDFVariableDeep
            var_ = cls(name=descr,
                       isolate=self._isolate,
                       timeaxis=self._timeaxis)
            self.variables[descr] = var_
        var_.log(sequence, infoarray)