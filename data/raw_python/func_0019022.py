def devices(self) -> selectiontools.Selection:
        """The additional devices defined for the respective `reader`
        or `writer` element contained within a |Selection| object. ToDo

        If the `reader` or `writer` element does not define its own additional
        devices, |XMLInterface.devices| of |XMLInterface| is used.

        >>> from hydpy.core.examples import prepare_full_example_1
        >>> prepare_full_example_1()

        >>> from hydpy import HydPy, TestIO, XMLInterface
        >>> hp = HydPy('LahnH')
        >>> with TestIO():
        ...     hp.prepare_network()
        ...     interface = XMLInterface('single_run.xml')
        >>> series_io = interface.series_io
        >>> for seq in (series_io.readers + series_io.writers):
        ...     print(seq.info, seq.devices.nodes, seq.devices.elements)
        all input data Nodes() \
Elements("land_dill", "land_lahn_1", "land_lahn_2", "land_lahn_3")
        precipitation Nodes() Elements("land_lahn_1", "land_lahn_2")
        soilmoisture Nodes("dill") Elements("land_dill", "land_lahn_1")
        averaged Nodes() Elements()
        """
        devices = self.find('devices')
        master = self
        while devices is None:
            master = master.master
            devices = master.find('devices')
        return _query_devices(devices)