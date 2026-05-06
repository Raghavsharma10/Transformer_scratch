def do_run(self):
        ''' Start runs on all modules sequentially.

        Sets properties to access current module properties.
        '''
        if self.broadcast_commands:  # Broadcast FE commands
            if self.threaded_scan:
                with ExitStack() as restore_config_stack:
                    # Configure each FE individually
                    # Sort module config keys, configure broadcast modules first
                    for module_id in itertools.chain(self._tx_module_groups, self._modules):
                        if self.abort_run.is_set():
                            break
                        with self.access_module(module_id=module_id):
                            if module_id in self._modules:
                                module_id_str = "module " + module_id
                            else:
                                module_id_str = module_id.split('=', 1)
                                module_id_str[0] = module_id_str[0].replace("_", " ")
                                module_id_str = "=".join(module_id_str)
                            logging.info('Scan parameter(s) for %s: %s', module_id_str, ', '.join(['%s=%s' % (key, value) for (key, value) in self.scan_parameters._asdict().items()]) if self.scan_parameters else 'None')
                            # storing register values until scan has finished and then restore configuration
                            restore_config_stack.enter_context(self.register.restored(name=self.run_number))
                            self.configure()
                    for module_id in self._tx_module_groups:
                        if self.abort_run.is_set():
                            break
                        with self.access_module(module_id=module_id):
                            # set all modules to run mode by before entering scan()
                            self.register_utils.set_run_mode()

                    with self.access_module(module_id=None):
                        self.fifo_readout.reset_rx()
                        self.fifo_readout.reset_fifo(self._selected_fifos)
                        self.fifo_readout.print_fei4_rx_status()

                        with self.access_files():
                            self._scan_threads = []
                            for module_id in self._tx_module_groups:
                                if self.abort_run.is_set():
                                    break
                                t = ExcThread(target=self.scan, name=module_id)
                                t.daemon = True  # exiting program even when thread is alive
                                self._scan_threads.append(t)
                            for t in self._scan_threads:
                                t.start()
                            while any([t.is_alive() for t in self._scan_threads]):
#                                 if self.abort_run.is_set():
#                                     break
                                for t in self._scan_threads:
                                    try:
                                        t.join(0.01)
                                    except Exception:
                                        self._scan_threads.remove(t)
                                        self.handle_err(sys.exc_info())
#                             alive_threads = [t.name for t in self._scan_threads if (not t.join(10.0) and t.is_alive())]
#                             if alive_threads:
#                                 raise RuntimeError("Scan thread(s) not finished: %s" % ", ".join(alive_threads))
                            self._scan_threads = []
                for module_id in self._tx_module_groups:
                    if self.abort_run.is_set():
                        break
                    with self.access_module(module_id=module_id):
                        # set modules to conf mode by after finishing scan()
                        self.register_utils.set_conf_mode()
            else:
                for tx_module_id, tx_group in self._tx_module_groups.items():
                    if self.abort_run.is_set():
                        break
                    with ExitStack() as restore_config_stack:
                        for module_id in itertools.chain([tx_module_id], tx_group):
                            if self.abort_run.is_set():
                                break
                            with self.access_module(module_id=module_id):
                                logging.info('Scan parameter(s) for module %s: %s', module_id, ', '.join(['%s=%s' % (key, value) for (key, value) in self.scan_parameters._asdict().items()]) if self.scan_parameters else 'None')
                                # storing register values until scan has finished and then restore configuration
                                restore_config_stack.enter_context(self.register.restored(name=self.run_number))
                                self.configure()
                        with self.access_module(module_id=tx_module_id):
                            # set all modules to run mode by before entering scan()
                            self.register_utils.set_run_mode()

                            self.fifo_readout.reset_rx()
                            self.fifo_readout.reset_fifo(self._selected_fifos)
                            self.fifo_readout.print_fei4_rx_status()

                            # some scans use this event to stop scan loop, clear event here to make another scan possible
                            self.stop_run.clear()
                            with self.access_files():
                                self.scan()

                    with self.access_module(module_id=tx_module_id):
                        # set modules to conf mode by after finishing scan()
                        self.register_utils.set_conf_mode()
        else:  # Scan each FE individually
            if self.threaded_scan:
                self._scan_threads = []
                # loop over grpups of modules with different TX
                for tx_module_ids in zip_nofill(*self._tx_module_groups.values()):
                    if self.abort_run.is_set():
                        break
                    with ExitStack() as restore_config_stack:
                        for module_id in tx_module_ids:
                            if self.abort_run.is_set():
                                break
                            with self.access_module(module_id=module_id):
                                logging.info('Scan parameter(s) for module %s: %s', module_id, ', '.join(['%s=%s' % (key, value) for (key, value) in self.scan_parameters._asdict().items()]) if self.scan_parameters else 'None')
                                # storing register values until scan has finished and then restore configuration
                                restore_config_stack.enter_context(self.register.restored(name=self.run_number))
                                self.configure()
                                # set modules to run mode by before entering scan()
                                self.register_utils.set_run_mode()
                            t = ExcThread(target=self.scan, name=module_id)
                            t.daemon = True  # exiting program even when thread is alive
                            self._scan_threads.append(t)
                        with self.access_module(module_id=tx_module_ids):
                            self.fifo_readout.reset_rx()
                            self.fifo_readout.reset_fifo(self._selected_fifos)
                            self.fifo_readout.print_fei4_rx_status()

                            with self.access_files():
                                # some scans use this event to stop scan loop, clear event here to make another scan possible
                                self.stop_run.clear()
                                for t in self._scan_threads:
                                    t.start()
                                while any([t.is_alive() for t in self._scan_threads]):
#                                     if self.abort_run.is_set():
#                                         break
                                    for t in self._scan_threads:
                                        try:
                                            t.join(0.01)
                                        except Exception:
                                            self._scan_threads.remove(t)
                                            self.handle_err(sys.exc_info())
#                                 alive_threads = [t.name for t in self._scan_threads if (not t.join(10.0) and t.is_alive())]
#                                 if alive_threads:
#                                     raise RuntimeError("Scan thread(s) not finished: %s" % ", ".join(alive_threads))
                                self._scan_threads = []

                    for module_id in tx_module_ids:
                        if self.abort_run.is_set():
                            break
                        with self.access_module(module_id=module_id):
                            # set modules to conf mode by after finishing scan()
                            self.register_utils.set_conf_mode()
            else:
                for module_id in self._modules:
                    if self.abort_run.is_set():
                        break
                    # some scans use this event to stop scan loop, clear event here to make another scan possible
                    self.stop_run.clear()
                    with self.access_module(module_id=module_id):
                        logging.info('Scan parameter(s) for module %s: %s', module_id, ', '.join(['%s=%s' % (key, value) for (key, value) in self.scan_parameters._asdict().items()]) if self.scan_parameters else 'None')
                        with self.register.restored(name=self.run_number):
                            self.configure()
                            # set modules to run mode by before entering scan()
                            self.register_utils.set_run_mode()

                            self.fifo_readout.reset_rx()
                            self.fifo_readout.reset_fifo(self._selected_fifos)
                            self.fifo_readout.print_fei4_rx_status()

                            # some scans use this event to stop scan loop, clear event here to make another scan possible
                            self.stop_run.clear()
                            with self.access_files():
                                self.scan()
                            # set modules to conf mode by after finishing scan()
                            self.register_utils.set_conf_mode()

        if self._modules:
            self.fifo_readout.print_readout_status()