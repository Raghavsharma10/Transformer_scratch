def item(self):
        """ ToDo

        >>> from hydpy.core.examples import prepare_full_example_1
        >>> prepare_full_example_1()

        >>> from hydpy import HydPy, TestIO, XMLInterface, pub
        >>> hp = HydPy('LahnH')
        >>> pub.timegrids = '1996-01-01', '1996-01-06', '1d'
        >>> with TestIO():
        ...     hp.prepare_everything()
        ...     interface = XMLInterface('multiple_runs.xml')

        >>> var = interface.exchange.itemgroups[0].models[0].subvars[0].vars[0]
        >>> item = var.item
        >>> item.value
        array(2.0)
        >>> hp.elements.land_dill.model.parameters.control.alpha
        alpha(1.0)
        >>> item.update_variables()
        >>> hp.elements.land_dill.model.parameters.control.alpha
        alpha(2.0)

        >>> var = interface.exchange.itemgroups[0].models[2].subvars[0].vars[0]
        >>> item = var.item
        >>> item.value
        array(5.0)
        >>> hp.elements.stream_dill_lahn_2.model.parameters.control.lag
        lag(0.0)
        >>> item.update_variables()
        >>> hp.elements.stream_dill_lahn_2.model.parameters.control.lag
        lag(5.0)

        >>> var = interface.exchange.itemgroups[1].models[0].subvars[0].vars[0]
        >>> item = var.item
        >>> item.name
        'sm_lahn_2'
        >>> item.value
        array(123.0)
        >>> hp.elements.land_lahn_2.model.sequences.states.sm
        sm(138.31396, 135.71124, 147.54968, 145.47142, 154.96405, 153.32805,
           160.91917, 159.62434, 165.65575, 164.63255)
        >>> item.update_variables()
        >>> hp.elements.land_lahn_2.model.sequences.states.sm
        sm(123.0, 123.0, 123.0, 123.0, 123.0, 123.0, 123.0, 123.0, 123.0, 123.0)

        >>> var = interface.exchange.itemgroups[1].models[0].subvars[0].vars[1]
        >>> item = var.item
        >>> item.name
        'sm_lahn_1'
        >>> item.value
        array([ 110.,  120.,  130.,  140.,  150.,  160.,  170.,  180.,  190.,
                200.,  210.,  220.,  230.])
        >>> hp.elements.land_lahn_1.model.sequences.states.sm
        sm(99.27505, 96.17726, 109.16576, 106.39745, 117.97304, 115.56252,
           125.81523, 123.73198, 132.80035, 130.91684, 138.95523, 137.25983,
           142.84148)
        >>> from hydpy import pub
        >>> with pub.options.warntrim(False):
        ...     item.update_variables()
        >>> hp.elements.land_lahn_1.model.sequences.states.sm
        sm(110.0, 120.0, 130.0, 140.0, 150.0, 160.0, 170.0, 180.0, 190.0, 200.0,
           206.0, 206.0, 206.0)

        >>> for element in pub.selections.headwaters.elements:
        ...     element.model.parameters.control.rfcf(1.1)
        >>> for element in pub.selections.nonheadwaters.elements:
        ...     element.model.parameters.control.rfcf(1.0)

        >>> for subvars in interface.exchange.itemgroups[2].models[0].subvars:
        ...     for var in subvars.vars:
        ...         var.item.update_variables()
        >>> for element in hp.elements.catchment:
        ...     print(element, repr(element.model.parameters.control.sfcf))
        land_dill sfcf(1.4)
        land_lahn_1 sfcf(1.4)
        land_lahn_2 sfcf(1.2)
        land_lahn_3 sfcf(field=1.1, forest=1.2)

        >>> var = interface.exchange.itemgroups[3].models[0].subvars[1].vars[0]
        >>> hp.elements.land_dill.model.sequences.states.sm = 1.0
        >>> for name, target in var.item.yield_name2value():
        ...     print(name, target)    # doctest: +ELLIPSIS
        land_dill_states_sm [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, \
1.0, 1.0, 1.0]
        land_lahn_1_states_sm [110.0, 120.0, 130.0, 140.0, 150.0, 160.0, \
170.0, 180.0, 190.0, 200.0, 206.0, 206.0, 206.0]
        land_lahn_2_states_sm [123.0, 123.0, 123.0, 123.0, 123.0, 123.0, \
123.0, 123.0, 123.0, 123.0]
        land_lahn_3_states_sm [101.3124...]

        >>> vars_ = interface.exchange.itemgroups[3].models[0].subvars[0].vars
        >>> qt = hp.elements.land_dill.model.sequences.fluxes.qt
        >>> qt(1.0)
        >>> qt.series = 2.0
        >>> for var in vars_:
        ...     for name, target in var.item.yield_name2value():
        ...         print(name, target)    # doctest: +ELLIPSIS
        land_dill_fluxes_qt 1.0
        land_dill_fluxes_qt_series [2.0, 2.0, 2.0, 2.0, 2.0]

        >>> var = interface.exchange.itemgroups[3].nodes[0].vars[0]
        >>> hp.nodes.dill.sequences.sim.series = range(5)
        >>> for name, target in var.item.yield_name2value():
        ...     print(name, target)    # doctest: +ELLIPSIS
        dill_nodes_sim_series [0.0, 1.0, 2.0, 3.0, 4.0]
        >>> for name, target in var.item.yield_name2value(2, 4):
        ...     print(name, target)    # doctest: +ELLIPSIS
        dill_nodes_sim_series [2.0, 3.0]
        """
        target = f'{self.master.name}.{self.name}'
        if self.master.name == 'nodes':
            master = self.master.name
            itemgroup = self.master.master.name
        else:
            master = self.master.master.name
            itemgroup = self.master.master.master.name
        itemclass = _ITEMGROUP2ITEMCLASS[itemgroup]
        if itemgroup == 'getitems':
            return self._get_getitem(target, master, itemclass)
        return self._get_changeitem(target, master, itemclass, itemgroup)