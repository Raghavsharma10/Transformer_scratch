def collect_variables(self, selections) -> None:
        """Apply method |ChangeItem.collect_variables| of the base class
        |ChangeItem| and also apply method |ExchangeItem.insert_variables|
        of class |ExchangeItem| to collect the relevant base variables
        handled by the devices of the given |Selections| object.

        >>> from hydpy.core.examples import prepare_full_example_2
        >>> hp, pub, TestIO = prepare_full_example_2()
        >>> from hydpy import AddItem
        >>> item = AddItem(
        ...     'alpha', 'hland_v1', 'control.sfcf', 'control.rfcf', 0)
        >>> item.collect_variables(pub.selections)
        >>> land_dill = hp.elements.land_dill
        >>> control = land_dill.model.parameters.control
        >>> item.device2target[land_dill] is control.sfcf
        True
        >>> item.device2base[land_dill] is control.rfcf
        True
        >>> for device in sorted(item.device2base, key=lambda x: x.name):
        ...     print(device)
        land_dill
        land_lahn_1
        land_lahn_2
        land_lahn_3
        """
        super().collect_variables(selections)
        self.insert_variables(self.device2base, self.basespecs, selections)