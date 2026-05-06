def _parse_module_cfgs(self):
        ''' Extracts the configuration of the modules.
        '''
        # Adding here default run config parameters.
        if "dut" not in self._conf or self._conf["dut"] is None:
            raise ValueError('Parameter "dut" not defined.')
        if "dut_configuration" not in self._conf or self._conf["dut_configuration"] is None:
            raise ValueError('Parameter "dut_configuration" not defined.')
        self._conf.setdefault('working_dir', None)  # string, if None, absolute path of configuration.yaml file will be used

        if 'modules' in self._conf and self._conf['modules']:
            for module_id, module_cfg in [(key, value) for key, value in self._conf['modules'].items() if ("activate" not in value or ("activate" in value and value["activate"] is True))]:
                # Check here for missing module config items.
                # Capital letter keys are Basil drivers, other keys are parameters.
                # FIFO, RX, TX, TLU and TDC are generic driver names which are used in the scan implementations.
                # The use of these reserved driver names allows for abstraction.
                # Accessing Basil drivers with real name is still possible.
                if "module_group" in module_id:
                    raise ValueError('The module ID "%s" contains the reserved name "module_group".' % module_id)
                if "flavor" not in module_cfg or module_cfg["flavor"] is None:
                    raise ValueError('No parameter "flavor" defined for module "%s".' % module_id)
                if module_cfg["flavor"] in fe_flavors:
                    for driver_name in _reserved_driver_names:
                        # TDC is not mandatory
                        if driver_name == "TDC":
                            # TDC is allowed to have set None
                            module_cfg.setdefault('TDC', None)
                            continue
                        if driver_name not in module_cfg or module_cfg[driver_name] is None:
                            raise ValueError('No parameter "%s" defined for module "%s".' % (driver_name, module_id))
                    if "rx_channel" not in module_cfg or module_cfg["rx_channel"] is None:
                        raise ValueError('No parameter "rx_channel" defined for module "%s".' % module_id)
                    if "tx_channel" not in module_cfg or module_cfg["tx_channel"] is None:
                        raise ValueError('No parameter "tx_channel" defined for module "%s".' % module_id)
                    if "chip_address" not in module_cfg:
                        raise ValueError('No parameter "chip_address" defined for module "%s".' % module_id)
                    module_cfg.setdefault("tdc_channel", None)
                    module_cfg.setdefault("configuration", None)  # string or number, if None, using the last valid configuration
                    module_cfg.setdefault("send_data", None)  # address string of PUB socket
                    module_cfg.setdefault("activate", True)  # set module active by default
                    # Save config to dict.
                    self._module_cfgs[module_id] = module_cfg
                    self._modules[module_id] = [module_id]
        else:
            raise ValueError("No module configuration specified")