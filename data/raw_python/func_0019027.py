def load_series(self) -> None:
        """Load time series data as defined by the actual XML `reader`
        element.

        >>> from hydpy.core.examples import prepare_full_example_1
        >>> prepare_full_example_1()

        >>> from hydpy import HydPy, TestIO, XMLInterface
        >>> hp = HydPy('LahnH')
        >>> with TestIO():
        ...     hp.prepare_network()
        ...     hp.init_models()
        ...     interface = XMLInterface('single_run.xml')
        ...     interface.update_options()
        ...     interface.update_timegrids()
        ...     series_io = interface.series_io
        ...     series_io.prepare_series()
        ...     series_io.load_series()
        >>> from hydpy import print_values
        >>> print_values(
        ...     hp.elements.land_dill.model.sequences.inputs.t.series[:3])
        -0.298846, -0.811539, -2.493848
        """
        kwargs = {}
        for keyword in ('flattennetcdf', 'isolatenetcdf', 'timeaxisnetcdf'):
            argument = getattr(hydpy.pub.options, keyword, None)
            if argument is not None:
                kwargs[keyword[:-6]] = argument
        hydpy.pub.sequencemanager.open_netcdf_reader(**kwargs)
        self.prepare_sequencemanager()
        for sequence in self._iterate_sequences():
            sequence.load_ext()
        hydpy.pub.sequencemanager.close_netcdf_reader()