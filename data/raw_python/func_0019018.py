def load_conditions(self) -> None:
        """Load the condition files of the |Model| objects of all |Element|
        objects returned by |XMLInterface.elements|:

        >>> from hydpy.core.examples import prepare_full_example_1
        >>> prepare_full_example_1()

        >>> from hydpy import HydPy, TestIO, XMLInterface
        >>> hp = HydPy('LahnH')
        >>> with TestIO():
        ...     hp.prepare_network()
        ...     hp.init_models()
        ...     interface = XMLInterface('single_run.xml')
        ...     interface.update_timegrids()
        ...     interface.find('selections').text = 'headwaters'
        ...     interface.conditions_io.load_conditions()
        >>> hp.elements.land_lahn_1.model.sequences.states.lz
        lz(8.18711)
        >>> hp.elements.land_lahn_2.model.sequences.states.lz
        lz(nan)
        """
        hydpy.pub.conditionmanager.currentdir = strip(
            self.find('inputdir').text)
        for element in self.master.elements:
            element.model.sequences.load_conditions()