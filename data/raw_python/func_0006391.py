def select_module(self, module_id):
        ''' Select module and give access to the module.
        '''
        if not isinstance(module_id, basestring) and isinstance(module_id, Iterable) and set(module_id) - set(self._modules):
            raise ValueError('Module IDs invalid:' % ", ".join(set(module_id) - set(self._modules)))
        if isinstance(module_id, basestring) and module_id not in self._module_cfgs:
            raise ValueError('Module ID "%s" is not valid' % module_id)
        if self._current_module_handle is not None:
            raise RuntimeError('Module handle "%s" cannot be set because another module is active' % module_id)

        if module_id is None:
            self._selected_modules = self._modules.keys()
        elif not isinstance(module_id, basestring) and isinstance(module_id, Iterable):
            self._selected_modules = module_id
        elif module_id in self._modules:
            self._selected_modules = [module_id]
        elif module_id in self._tx_module_groups:
            self._selected_modules = self._tx_module_groups[module_id]
        else:
            RuntimeError('Cannot open files. Module handle "%s" is not valid.' % self.current_module_handle)

        # FIFO readout
        self._selected_fifos = list(set([module_cfg['FIFO'] for (name, module_cfg) in self._module_cfgs.items() if name in self._selected_modules]))

        # Module filter functions dict for quick lookup
        self._readout_fifos = []
        self._filter = []
        self._converter = []
        for selected_module_id in self._selected_modules:
            module_cfg = self._module_cfgs[selected_module_id]
            self._readout_fifos.append(module_cfg['FIFO'])
            if 'tdc_channel' not in module_cfg:
                tdc_filter = false
                self._converter.append(None)
            elif module_cfg['tdc_channel'] is None:
                tdc_filter = is_tdc_word
                self._converter.append(convert_tdc_to_channel(channel=module_cfg['tdc_channel']))  # for the raw data analyzer
            else:
                tdc_filter = logical_and(is_tdc_word, is_tdc_from_channel(module_cfg['tdc_channel']))
                self._converter.append(convert_tdc_to_channel(channel=module_cfg['tdc_channel']))  # for the raw data analyzer
            if 'rx_channel' not in module_cfg:
                self._filter.append(logical_or(is_trigger_word, tdc_filter))
            elif module_cfg['rx_channel'] is None:
                self._filter.append(logical_or(is_trigger_word, logical_or(tdc_filter, is_fe_word)))
            else:
                self._filter.append(logical_or(is_trigger_word, logical_or(tdc_filter, logical_and(is_fe_word, is_data_from_channel(module_cfg['rx_channel'])))))

        # select readout channels and report sync status only from actively selected modules
        self._enabled_fe_channels = list(set([config['RX'] for (name, config) in self._module_cfgs.items() if name in self._selected_modules]))

        # enabling specific TX channels
        tx_channels = list(set([1 << config['tx_channel'] for (name, config) in self._module_cfgs.items() if name in self._selected_modules]))
        if tx_channels:
            self.dut['TX']['OUTPUT_ENABLE'] = reduce(lambda x, y: x | y, tx_channels)
        else:
            self.dut['TX']['OUTPUT_ENABLE'] = 0

        if not isinstance(module_id, basestring) and isinstance(module_id, Iterable):
            self._current_module_handle = None
        else:
            self._current_module_handle = module_id

        if module_id is not None and isinstance(module_id, basestring):
            current_thread().name = module_id