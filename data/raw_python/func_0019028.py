def save_series(self) -> None:
        """Save time series data as defined by the actual XML `writer`
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
        >>> interface.update_timegrids()
        >>> series_io = interface.series_io
        >>> series_io.prepare_series()
        >>> hp.elements.land_dill.model.sequences.fluxes.pc.series[2, 3] = 9.0
        >>> hp.nodes.lahn_2.sequences.sim.series[4] = 7.0
        >>> with TestIO():
        ...     series_io.save_series()
        >>> import numpy
        >>> with TestIO():
        ...     os.path.exists(
        ...         'LahnH/series/output/land_lahn_2_flux_pc.npy')
        ...     os.path.exists(
        ...         'LahnH/series/output/land_lahn_3_flux_pc.npy')
        ...     numpy.load(
        ...         'LahnH/series/output/land_dill_flux_pc.npy')[13+2, 3]
        ...     numpy.load(
        ...         'LahnH/series/output/lahn_2_sim_q_mean.npy')[13+4]
        True
        False
        9.0
        7.0
        """
        hydpy.pub.sequencemanager.open_netcdf_writer(
            flatten=hydpy.pub.options.flattennetcdf,
            isolate=hydpy.pub.options.isolatenetcdf)
        self.prepare_sequencemanager()
        for sequence in self._iterate_sequences():
            sequence.save_ext()
        hydpy.pub.sequencemanager.close_netcdf_writer()