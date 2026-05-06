def prepare_series(self) -> None:
        # noinspection PyUnresolvedReferences
        """Call |XMLSubseries.prepare_series| of all |XMLSubseries|
        objects with the same memory |set| object.

        >>> from hydpy.auxs.xmltools import XMLInterface, XMLSubseries
        >>> from hydpy import data
        >>> interface = XMLInterface('single_run.xml', data.get_path('LahnH'))
        >>> series_io = interface.series_io
        >>> from unittest import mock
        >>> prepare_series = XMLSubseries.prepare_series
        >>> XMLSubseries.prepare_series = mock.MagicMock()
        >>> series_io.prepare_series()
        >>> args = XMLSubseries.prepare_series.call_args_list
        >>> len(args) == len(series_io.readers) + len(series_io.writers)
        True
        >>> args[0][0][0]
        set()
        >>> args[0][0][0] is args[-1][0][0]
        True
        >>> XMLSubseries.prepare_series = prepare_series
        """
        memory = set()
        for output in itertools.chain(self.readers, self.writers):
            output.prepare_series(memory)