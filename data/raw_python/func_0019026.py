def prepare_series(self, memory: set) -> None:
        """Call |IOSequence.activate_ram| of all sequences selected by
        the given output element of the actual XML file.

        Use the memory argument to pass in already prepared sequences;
        newly prepared sequences will be added.

        >>> from hydpy.core.examples import prepare_full_example_1
        >>> prepare_full_example_1()

        >>> from hydpy import HydPy, TestIO, XMLInterface
        >>> hp = HydPy('LahnH')
        >>> with TestIO():
        ...     hp.prepare_network()
        ...     hp.init_models()
        ...     interface = XMLInterface('single_run.xml')
        >>> interface.update_timegrids()
        >>> series_io = interface.series_io

        >>> memory = set()
        >>> pc = hp.elements.land_dill.model.sequences.fluxes.pc
        >>> pc.ramflag
        False
        >>> series_io.writers[0].prepare_series(memory)
        >>> pc in memory
        True
        >>> pc.ramflag
        True

        >>> pc.deactivate_ram()
        >>> pc.ramflag
        False
        >>> series_io.writers[0].prepare_series(memory)
        >>> pc.ramflag
        False
        """
        for sequence in self._iterate_sequences():
            if sequence not in memory:
                memory.add(sequence)
                sequence.activate_ram()