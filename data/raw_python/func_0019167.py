def yield_name2value(self, idx1=None, idx2=None) \
            -> Iterator[Tuple[str, str]]:
        """Sequentially return name-value-pairs describing the current state
        of the target variables.

        The names are automatically generated and contain both the name of
        the |Device| of the respective |Variable| object and the target
        description:

        >>> from hydpy.core.examples import prepare_full_example_2
        >>> hp, pub, TestIO = prepare_full_example_2()
        >>> from hydpy.core.itemtools import SetItem
        >>> item = GetItem('hland_v1', 'states.lz')
        >>> item.collect_variables(pub.selections)
        >>> hp.elements.land_dill.model.sequences.states.lz = 100.0
        >>> for name, value in item.yield_name2value():
        ...     print(name, value)
        land_dill_states_lz 100.0
        land_lahn_1_states_lz 8.18711
        land_lahn_2_states_lz 10.14007
        land_lahn_3_states_lz 7.52648
        >>> item = GetItem('hland_v1', 'states.sm')
        >>> item.collect_variables(pub.selections)
        >>> hp.elements.land_dill.model.sequences.states.sm = 2.0
        >>> for name, value in item.yield_name2value():
        ...     print(name, value)    # doctest: +ELLIPSIS
        land_dill_states_sm [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, \
2.0, 2.0, 2.0, 2.0]
        land_lahn_1_states_sm [99.27505, ..., 142.84148]
        ...

        When querying time series, one can restrict the span of interest
        by passing index values:

        >>> item = GetItem('nodes', 'sim.series')
        >>> item.collect_variables(pub.selections)
        >>> hp.nodes.dill.sequences.sim.series = 1.0, 2.0, 3.0, 4.0
        >>> for name, value in item.yield_name2value():
        ...     print(name, value)    # doctest: +ELLIPSIS
        dill_sim_series [1.0, 2.0, 3.0, 4.0]
        lahn_1_sim_series [nan, ...
        ...
        >>> for name, value in item.yield_name2value(2, 3):
        ...     print(name, value)    # doctest: +ELLIPSIS
        dill_sim_series [3.0]
        lahn_1_sim_series [nan]
        ...
        """
        for device, name in self._device2name.items():
            target = self.device2target[device]
            if self.targetspecs.series:
                values = target.series[idx1:idx2]
            else:
                values = target.values
            if self.ndim == 0:
                values = objecttools.repr_(float(values))
            else:
                values = objecttools.repr_list(values.tolist())
            yield name, values