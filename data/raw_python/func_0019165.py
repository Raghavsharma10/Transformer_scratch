def update_variables(self) -> None:
        """Add the general |ChangeItem.value| with the |Device| specific base
        variable and assign the result to the respective target variable.

        >>> from hydpy.core.examples import prepare_full_example_2
        >>> hp, pub, TestIO = prepare_full_example_2()
        >>> from hydpy.models.hland_v1 import FIELD
        >>> for element in hp.elements.catchment:
        ...     control = element.model.parameters.control
        ...     control.nmbzones(3)
        ...     control.zonetype(FIELD)
        ...     control.rfcf(1.1)
        >>> from hydpy.core.itemtools import AddItem
        >>> item = AddItem(
        ...     'sfcf', 'hland_v1', 'control.sfcf', 'control.rfcf', 1)
        >>> item.collect_variables(pub.selections)
        >>> land_dill = hp.elements.land_dill
        >>> land_dill.model.parameters.control.sfcf
        sfcf(?)
        >>> item.value = -0.1, 0.0, 0.1
        >>> item.update_variables()
        >>> land_dill.model.parameters.control.sfcf
        sfcf(1.0, 1.1, 1.2)

        >>> land_dill.model.parameters.control.rfcf.shape = 2
        >>> land_dill.model.parameters.control.rfcf = 1.1
        >>> item.update_variables()    # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        ValueError: When trying to add the value(s) `[-0.1  0.   0.1]` of \
AddItem `sfcf` and the value(s) `[ 1.1  1.1]` of variable `rfcf` of element \
`land_dill`, the following error occurred: operands could not be broadcast \
together with shapes (2,) (3,)...
        """
        value = self.value
        for device, target in self.device2target.items():
            base = self.device2base[device]
            try:
                result = base.value + value
            except BaseException:
                raise objecttools.augment_excmessage(
                    f'When trying to add the value(s) `{value}` of '
                    f'AddItem `{self.name}` and the value(s) `{base.value}` '
                    f'of variable {objecttools.devicephrase(base)}')
            self.update_variable(target, result)