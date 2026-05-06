def collect_variables(self, selections) -> None:
        """Apply method |ExchangeItem.collect_variables| of the base class
        |ExchangeItem| and determine the `ndim` attribute of the current
        |ChangeItem| object afterwards.

        The value of `ndim` depends on whether the values of the target
        variable or its time series is of interest:

        >>> from hydpy.core.examples import prepare_full_example_2
        >>> hp, pub, TestIO = prepare_full_example_2()
        >>> from hydpy.core.itemtools import SetItem
        >>> for target in ('states.lz', 'states.lz.series',
        ...                'states.sm', 'states.sm.series'):
        ...     item = GetItem('hland_v1', target)
        ...     item.collect_variables(pub.selections)
        ...     print(item, item.ndim)
        GetItem('hland_v1', 'states.lz') 0
        GetItem('hland_v1', 'states.lz.series') 1
        GetItem('hland_v1', 'states.sm') 1
        GetItem('hland_v1', 'states.sm.series') 2
        """
        super().collect_variables(selections)
        for device in sorted(self.device2target.keys(), key=lambda x: x.name):
            self._device2name[device] = f'{device.name}_{self.target}'
        for target in self.device2target.values():
            self.ndim = target.NDIM
            if self.targetspecs.series:
                self.ndim += 1
            break