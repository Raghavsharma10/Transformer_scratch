def elements(self) -> Iterator[devicetools.Element]:
        """Yield all |Element| objects returned by |XMLInterface.selections|
        and |XMLInterface.devices| without duplicates.

        >>> from hydpy.core.examples import prepare_full_example_1
        >>> prepare_full_example_1()

        >>> from hydpy import HydPy, TestIO, XMLInterface
        >>> hp = HydPy('LahnH')
        >>> with TestIO():
        ...     hp.prepare_network()
        ...     interface = XMLInterface('single_run.xml')
        >>> interface.find('selections').text = 'headwaters streams'
        >>> for element in interface.elements:
        ...      print(element.name)
        land_dill
        land_lahn_1
        stream_dill_lahn_2
        stream_lahn_1_lahn_2
        stream_lahn_2_lahn_3
        """
        selections = copy.copy(self.selections)
        selections += self.devices
        elements = set()
        for selection in selections:
            for element in selection.elements:
                if element not in elements:
                    elements.add(element)
                    yield element