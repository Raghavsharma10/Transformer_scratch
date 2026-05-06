def save_conditions(self) -> None:
        """Save the condition files of the |Model| objects of all |Element|
        objects returned by |XMLInterface.elements|:

        >>> from hydpy.core.examples import prepare_full_example_1
        >>> prepare_full_example_1()

        >>> import os
        >>> from hydpy import HydPy, TestIO, XMLInterface, pub
        >>> hp = HydPy('LahnH')
        >>> with TestIO():
        ...     hp.prepare_network()
        ...     hp.init_models()
        ...     hp.elements.land_dill.model.sequences.states.lz = 999.0
        ...     interface = XMLInterface('single_run.xml')
        ...     interface.update_timegrids()
        ...     interface.find('selections').text = 'headwaters'
        ...     interface.conditions_io.save_conditions()
        ...     dirpath = 'LahnH/conditions/init_1996_01_06'
        ...     with open(os.path.join(dirpath, 'land_dill.py')) as file_:
        ...         print(file_.readlines()[11].strip())
        ...     os.path.exists(os.path.join(dirpath, 'land_lahn_2.py'))
        lz(999.0)
        False
        """
        hydpy.pub.conditionmanager.currentdir = strip(
            self.find('outputdir').text)
        for element in self.master.elements:
            element.model.sequences.save_conditions()
        if strip(self.find('zip').text) == 'true':
            hydpy.pub.conditionmanager.zip_currentdir()