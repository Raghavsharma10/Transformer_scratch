def collect_variables(self, selections) -> None:
        """Apply method |ExchangeItem.insert_variables| to collect the
        relevant target variables handled by the devices of the given
        |Selections| object.

        We prepare the `LahnH` example project to be able to use its
        |Selections| object:

        >>> from hydpy.core.examples import prepare_full_example_2
        >>> hp, pub, TestIO = prepare_full_example_2()

        We change the type of a specific application model to the type
        of its base model for reasons explained later:

        >>> from hydpy.models.hland import Model
        >>> hp.elements.land_lahn_3.model.__class__ = Model

        We prepare a |SetItem| as an example, handling all |hland_states.Ic|
        sequences corresponding to any application models derived from |hland|:

        >>> from hydpy import SetItem
        >>> item = SetItem('ic', 'hland', 'states.ic', 0)
        >>> item.targetspecs
        ExchangeSpecification('hland', 'states.ic')

        Applying method |ExchangeItem.collect_variables| connects the |SetItem|
        object with all four relevant |hland_states.Ic| objects:

        >>> item.collect_variables(pub.selections)
        >>> land_dill = hp.elements.land_dill
        >>> sequence = land_dill.model.sequences.states.ic
        >>> item.device2target[land_dill] is sequence
        True
        >>> for element in sorted(item.device2target, key=lambda x: x.name):
        ...     print(element)
        land_dill
        land_lahn_1
        land_lahn_2
        land_lahn_3

        Asking for |hland_states.Ic| objects corresponding to application
        model |hland_v1| only, results in skipping the |Element| `land_lahn_3`
        (handling the |hland| base model due to the hack above):

        >>> item = SetItem('ic', 'hland_v1', 'states.ic', 0)
        >>> item.collect_variables(pub.selections)
        >>> for element in sorted(item.device2target, key=lambda x: x.name):
        ...     print(element)
        land_dill
        land_lahn_1
        land_lahn_2

        Selecting a series of a variable instead of the variable itself
        only affects the `targetspec` attribute:

        >>> item = SetItem('t', 'hland_v1', 'inputs.t.series', 0)
        >>> item.collect_variables(pub.selections)
        >>> item.targetspecs
        ExchangeSpecification('hland_v1', 'inputs.t.series')
        >>> sequence = land_dill.model.sequences.inputs.t
        >>> item.device2target[land_dill] is sequence
        True

        It is both possible to address sequences of |Node| objects, as well
        as their time series, by arguments "node" and "nodes":

        >>> item = SetItem('sim', 'node', 'sim', 0)
        >>> item.collect_variables(pub.selections)
        >>> dill = hp.nodes.dill
        >>> item.targetspecs
        ExchangeSpecification('node', 'sim')
        >>> item.device2target[dill] is dill.sequences.sim
        True
        >>> for node in sorted(item.device2target, key=lambda x: x.name):
        ...  print(node)
        dill
        lahn_1
        lahn_2
        lahn_3
        >>> item = SetItem('sim', 'nodes', 'sim.series', 0)
        >>> item.collect_variables(pub.selections)
        >>> item.targetspecs
        ExchangeSpecification('nodes', 'sim.series')
        >>> for node in sorted(item.device2target, key=lambda x: x.name):
        ...  print(node)
        dill
        lahn_1
        lahn_2
        lahn_3
        """
        self.insert_variables(self.device2target, self.targetspecs, selections)