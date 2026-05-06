def aggregate_series(self, *args, **kwargs) -> InfoArray:
        """Aggregates time series data based on the actual
        |FluxSequence.aggregation_ext| attribute of |IOSequence|
        subclasses.

        We prepare some nodes and elements with the help of
        method |prepare_io_example_1| and select a 1-dimensional
        flux sequence of type |lland_fluxes.NKor| as an example:

        >>> from hydpy.core.examples import prepare_io_example_1
        >>> nodes, elements = prepare_io_example_1()
        >>> seq = elements.element3.model.sequences.fluxes.nkor

        If no |FluxSequence.aggregation_ext| is `none`, the
        original time series values are returned:

        >>> seq.aggregation_ext
        'none'
        >>> seq.aggregate_series()
        InfoArray([[ 24.,  25.,  26.],
                   [ 27.,  28.,  29.],
                   [ 30.,  31.,  32.],
                   [ 33.,  34.,  35.]])

        If no |FluxSequence.aggregation_ext| is `mean`, function
        |IOSequence.aggregate_series| is called:

        >>> seq.aggregation_ext = 'mean'
        >>> seq.aggregate_series()
        InfoArray([ 25.,  28.,  31.,  34.])

        In case the state of the sequence is invalid:

        >>> seq.aggregation_ext = 'nonexistent'
        >>> seq.aggregate_series()
        Traceback (most recent call last):
        ...
        RuntimeError: Unknown aggregation mode `nonexistent` for \
sequence `nkor` of element `element3`.

        The following technical test confirms that all potential
        positional and keyword arguments are passed properly:
        >>> seq.aggregation_ext = 'mean'

        >>> from unittest import mock
        >>> seq.average_series = mock.MagicMock()
        >>> _ = seq.aggregate_series(1, x=2)
        >>> seq.average_series.assert_called_with(1, x=2)
        """
        mode = self.aggregation_ext
        if mode == 'none':
            return self.series
        elif mode == 'mean':
            return self.average_series(*args, **kwargs)
        else:
            raise RuntimeError(
                'Unknown aggregation mode `%s` for sequence %s.'
                % (mode, objecttools.devicephrase(self)))