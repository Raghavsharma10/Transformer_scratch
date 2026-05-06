def selections(self) -> selectiontools.Selections:
        """The |Selections| object defined for the respective `reader`
        or `writer` element of the actual XML file. ToDo

        If the `reader` or `writer` element does not define a special
        selections element, the general |XMLInterface.selections| element
        of |XMLInterface| is used.

        >>> from hydpy.core.examples import prepare_full_example_1
        >>> prepare_full_example_1()

        >>> from hydpy import HydPy, TestIO, XMLInterface
        >>> hp = HydPy('LahnH')
        >>> with TestIO():
        ...     hp.prepare_network()
        ...     hp.init_models()
        ...     interface = XMLInterface('single_run.xml')
        >>> series_io = interface.series_io
        >>> for seq in (series_io.readers + series_io.writers):
        ...     print(seq.info, seq.selections.names)
        all input data ()
        precipitation ('headwaters',)
        soilmoisture ('complete',)
        averaged ('complete',)
        """
        selections = self.find('selections')
        master = self
        while selections is None:
            master = master.master
            selections = master.find('selections')
        return _query_selections(selections)