def _set_default_cfg(self):
        ''' Sets the default parameters if they are not specified.
        '''
        # adding special conf for accessing all DUT drivers
        self._module_cfgs[None] = {
            'flavor': None,
            'chip_address': None,
            'FIFO': list(set([self._module_cfgs[module_id]['FIFO'] for module_id in self._modules])),
            'RX': list(set([self._module_cfgs[module_id]['RX'] for module_id in self._modules])),
            'rx_channel': list(set([self._module_cfgs[module_id]['rx_channel'] for module_id in self._modules])),
            'TX': list(set([self._module_cfgs[module_id]['TX'] for module_id in self._modules])),
            'tx_channel': list(set([self._module_cfgs[module_id]['tx_channel'] for module_id in self._modules])),
            'TDC': list(set([self._module_cfgs[module_id]['TDC'] for module_id in self._modules])),
            'tdc_channel': list(set([self._module_cfgs[module_id]['tdc_channel'] for module_id in self._modules])),
            'TLU': list(set([self._module_cfgs[module_id]['TLU'] for module_id in self._modules])),
            'configuration': None,
            'send_data': None}

        tx_groups = groupby_dict({key: value for (key, value) in self._module_cfgs.items() if key in self._modules}, "TX")
        for tx, module_group in tx_groups.items():
            flavors = list(set([module_cfg['flavor'] for module_id, module_cfg in self._module_cfgs.items() if module_id in module_group]))
            if len(flavors) != 1:
                raise ValueError("Parameter 'flavor' must be the same for module group TX=%s." % tx)

            chip_addresses = list(set([module_cfg['chip_address'] for module_id, module_cfg in self._module_cfgs.items() if module_id in module_group]))
            if len(module_group) != len(chip_addresses) or (len(module_group) != 1 and None in chip_addresses):
                raise ValueError("Parameter 'chip_address' must be different for each module in module group TX=%s." % tx)

            # Adding broadcast config for parallel mode.
            self._module_cfgs["module_group_TX=" + tx] = {
                'flavor': flavors[0],
                'chip_address': None,  # broadcast
                'FIFO': list(set([module_cfg['FIFO'] for module_id, module_cfg in self._module_cfgs.items() if module_id in module_group])),
                'RX': list(set([module_cfg['RX'] for module_id, module_cfg in self._module_cfgs.items() if module_id in module_group])),
                'rx_channel': list(set([module_cfg['rx_channel'] for module_id, module_cfg in self._module_cfgs.items() if module_id in module_group])),
                'TX': tx,
                'tx_channel': list(set([module_cfg['tx_channel'] for module_id, module_cfg in self._module_cfgs.items() if module_id in module_group])),
                'TDC': list(set([module_cfg['TDC'] for module_id, module_cfg in self._module_cfgs.items() if module_id in module_group])),
                'tdc_channel': list(set([module_cfg['tdc_channel'] for module_id, module_cfg in self._module_cfgs.items() if module_id in module_group])),
                'TLU': list(set([module_cfg['TLU'] for module_id, module_cfg in self._module_cfgs.items() if module_id in module_group])),
                'configuration': None,
                'send_data': None}
            self._tx_module_groups["module_group_TX=" + tx] = module_group

        # Setting up per module attributes
        self._module_attr = {key: {} for key in self._module_cfgs}
        # Setting up per module run conf
        for module_id in self._module_cfgs:
            sc = namedtuple('run_configuration', field_names=self._default_run_conf.keys())
            run_conf = sc(**self._run_conf)
            if module_id in self._modules and self.__class__.__name__ in self._conf["modules"][module_id] and self._conf["modules"][module_id][self.__class__.__name__] is not None:
                self._module_run_conf[module_id] = run_conf._replace(**self._conf["modules"][module_id][self.__class__.__name__])._asdict()
            else:
                self._module_run_conf[module_id] = run_conf._asdict()
                # update module group with run specific configuration
                if module_id in self._tx_module_groups and self._tx_module_groups[module_id]:
                    selected_module_id = self._tx_module_groups[module_id][0]
                    if self.__class__.__name__ in self._conf["modules"][selected_module_id] and self._conf["modules"][selected_module_id][self.__class__.__name__] is not None:
                        self._module_run_conf[module_id] = run_conf._replace(**self._conf["modules"][selected_module_id][self.__class__.__name__])._asdict()