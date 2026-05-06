def update_variables(self) -> None:
        """Assign the current objects |ChangeItem.value| to the values
        of the target variables.

        We use the `LahnH` project in the following:

        >>> from hydpy.core.examples import prepare_full_example_2
        >>> hp, pub, TestIO = prepare_full_example_2()

        In the first example, a 0-dimensional |SetItem| changes the value
        of the 0-dimensional parameter |hland_control.Alpha|:

        >>> from hydpy.core.itemtools import SetItem
        >>> item = SetItem('alpha', 'hland_v1', 'control.alpha', 0)
        >>> item
        SetItem('alpha', 'hland_v1', 'control.alpha', 0)
        >>> item.collect_variables(pub.selections)
        >>> item.value is None
        True
        >>> land_dill = hp.elements.land_dill
        >>> land_dill.model.parameters.control.alpha
        alpha(1.0)
        >>> item.value = 2.0
        >>> item.value
        array(2.0)
        >>> land_dill.model.parameters.control.alpha
        alpha(1.0)
        >>> item.update_variables()
        >>> land_dill.model.parameters.control.alpha
        alpha(2.0)

        In the second example, a 0-dimensional |SetItem| changes the values
        of the 1-dimensional parameter |hland_control.FC|:


        >>> item = SetItem('fc', 'hland_v1', 'control.fc', 0)
        >>> item.collect_variables(pub.selections)
        >>> item.value = 200.0
        >>> land_dill.model.parameters.control.fc
        fc(278.0)
        >>> item.update_variables()
        >>> land_dill.model.parameters.control.fc
        fc(200.0)

        In the third example, a 1-dimensional |SetItem| changes the values
        of the 1-dimensional sequence |hland_states.Ic|:

        >>> for element in hp.elements.catchment:
        ...     element.model.parameters.control.nmbzones(5)
        ...     element.model.parameters.control.icmax(4.0)
        >>> item = SetItem('ic', 'hland_v1', 'states.ic', 1)
        >>> item.collect_variables(pub.selections)
        >>> land_dill.model.sequences.states.ic
        ic(nan, nan, nan, nan, nan)
        >>> item.value = 2.0
        >>> item.update_variables()
        >>> land_dill.model.sequences.states.ic
        ic(2.0, 2.0, 2.0, 2.0, 2.0)
        >>> item.value = 1.0, 2.0, 3.0, 4.0, 5.0
        >>> item.update_variables()
        >>> land_dill.model.sequences.states.ic
        ic(1.0, 2.0, 3.0, 4.0, 4.0)
        """
        value = self.value
        for variable in self.device2target.values():
            self.update_variable(variable, value)