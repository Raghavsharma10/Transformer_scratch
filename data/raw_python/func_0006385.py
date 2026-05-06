def init_modules(self):
        ''' Initialize all modules consecutively'''
        for module_id, module_cfg in self._module_cfgs.items():
            if module_id in self._modules or module_id in self._tx_module_groups:
                if module_id in self._modules:
                    module_id_str = "module " + module_id
                else:
                    module_id_str = module_id.split('=', 1)
                    module_id_str[0] = module_id_str[0].replace("_", " ")
                    module_id_str = "=".join(module_id_str)
                logging.info("Initializing configuration for %s..." % module_id_str)
                # adding scan parameters to dict
                if 'scan_parameters' in self._module_run_conf[module_id] and self._module_run_conf[module_id]['scan_parameters'] is not None:
                    # evaluating string for support of nested lists and other complex data structures
                    if isinstance(self._module_run_conf[module_id]['scan_parameters'], basestring):
                        self._module_run_conf[module_id]['scan_parameters'] = ast.literal_eval(self._module_run_conf[module_id]['scan_parameters'])
                    sp = namedtuple('scan_parameters', field_names=zip(*self._module_run_conf[module_id]['scan_parameters'])[0])
                    self._scan_parameters[module_id] = sp(*zip(*self._module_run_conf[module_id]['scan_parameters'])[1])
                else:
                    sp = namedtuple_with_defaults('scan_parameters', field_names=[])
                    self._scan_parameters[module_id] = sp()
                # init FE config
                if module_id in self._modules:
                    # only real modules can have an existing configuration
                    last_configuration = self.get_configuration(module_id=module_id)
                else:
                    last_configuration = None
                if (('configuration' not in module_cfg or module_cfg['configuration'] is None) and last_configuration is None) or (isinstance(module_cfg['configuration'], (int, long)) and module_cfg['configuration'] <= 0):
                    if 'chip_address' in module_cfg:
                        if module_cfg['chip_address'] is None:
                            chip_address = 0
                            broadcast = True
                        else:
                            chip_address = module_cfg['chip_address']
                            broadcast = False
                    else:
                        raise ValueError('Parameter "chip_address" not specified for module "%s".' % module_id)
                    if 'flavor' in module_cfg and module_cfg['flavor']:
                        module_cfg['configuration'] = FEI4Register(fe_type=module_cfg['flavor'], chip_address=chip_address, broadcast=broadcast)
                    else:
                        raise ValueError('Parameter "flavor" not specified for module "%s".' % module_id)
                # use existing config
                elif not module_cfg['configuration'] and last_configuration:
                    module_cfg['configuration'] = FEI4Register(configuration_file=last_configuration)
                # path string
                elif isinstance(module_cfg['configuration'], basestring):
                    if os.path.isabs(module_cfg['configuration']):  # absolute path
                        module_cfg['configuration'] = FEI4Register(configuration_file=module_cfg['configuration'])
                    else:  # relative path
                        module_cfg['configuration'] = FEI4Register(configuration_file=os.path.join(module_cfg['working_dir'], module_cfg['configuration']))
                # run number
                elif isinstance(module_cfg['configuration'], (int, long)) and module_cfg['configuration'] > 0:
                    module_cfg['configuration'] = FEI4Register(configuration_file=self.get_configuration(module_id=module_id, run_number=module_cfg['configuration']))
                # assume configuration already initialized
                elif not isinstance(module_cfg['configuration'], FEI4Register):
                    raise ValueError('Found no valid value for parameter "configuration" for module "%s".' % module_id)

                # init register utils
                self._registers[module_id] = self._module_cfgs[module_id]['configuration']
                self._register_utils[module_id] = FEI4RegisterUtils(self._module_dut[module_id], self._module_cfgs[module_id]['configuration'])

                if module_id in self._modules:
                    # Create module data path for real modules
                    module_path = self.get_module_path(module_id)
                    if not os.path.exists(module_path):
                        os.makedirs(module_path)

        # Set all modules to conf mode to prevent from receiving BCR and ECR broadcast
        for module_id in self._tx_module_groups:
            with self.access_module(module_id=module_id):
                self.register_utils.set_conf_mode()

        # Initial configuration (reset and configuration) of all modules.
        # This is done by iterating over each module individually
        for module_id in self._modules:
            logging.info("Configuring %s..." % module_id)
            with self.access_module(module_id=module_id):
                if self._run_conf['configure_fe']:
                    self.register_utils.global_reset()
                    self.register_utils.configure_all()
                else:
                    self.register_utils.set_conf_mode()
                if is_fe_ready(self):
                    fe_not_ready = False
                else:
                    fe_not_ready = True
                # BCR and ECR might result in RX errors
                # a reset of the RX and FIFO will happen just before scan()
                if self._run_conf['reset_fe']:
                    self.register_utils.reset_bunch_counter()
                    self.register_utils.reset_event_counter()
                if fe_not_ready:
                    # resetting service records must be done once after power up
                    self.register_utils.reset_service_records()
                    if not is_fe_ready(self):
                        logging.warning('Module "%s" is not sending any data.' % module_id)
                # set all modules to conf mode afterwards to be immune to ECR and BCR
                self.register_utils.set_conf_mode()