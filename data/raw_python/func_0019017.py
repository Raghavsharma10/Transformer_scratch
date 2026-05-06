def fullselection(self) -> selectiontools.Selection:
        """A |Selection| object containing all |Element| and |Node| objects
        defined by |XMLInterface.selections| and |XMLInterface.devices|.

        >>> from hydpy.core.examples import prepare_full_example_1
        >>> prepare_full_example_1()

        >>> from hydpy import HydPy, TestIO, XMLInterface
        >>> hp = HydPy('LahnH')
        >>> with TestIO():
        ...     hp.prepare_network()
        ...     interface = XMLInterface('single_run.xml')
        >>> interface.find('selections').text = 'nonheadwaters'
        >>> interface.fullselection
        Selection("fullselection",
                  nodes=("dill", "lahn_2", "lahn_3"),
                  elements=("land_dill", "land_lahn_1", "land_lahn_2",
                            "land_lahn_3"))
        """
        fullselection = selectiontools.Selection('fullselection')
        for selection in self.selections:
            fullselection += selection
        fullselection += self.devices
        return fullselection