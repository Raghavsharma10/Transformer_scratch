def start_readout(self, *args, **kwargs):
        ''' Starting the FIFO readout.

        Starting of the FIFO readout is executed only once by a random thread.
        Starting of the FIFO readout is synchronized between all threads reading out the FIFO.
        '''
        # Pop parameters for fifo_readout.start
        callback = kwargs.pop('callback', self.handle_data)
        errback = kwargs.pop('errback', self.handle_err)
        reset_rx = kwargs.pop('reset_rx', True)
        reset_fifo = kwargs.pop('reset_fifo', True)
        fill_buffer = kwargs.pop('fill_buffer', False)
        no_data_timeout = kwargs.pop('no_data_timeout', None)
        enabled_fe_channels = kwargs.pop('enabled_fe_channels', self._enabled_fe_channels)
        if args or kwargs:
            self.set_scan_parameters(*args, **kwargs)
        if self._scan_threads and self.current_module_handle not in [t.name for t in self._scan_threads]:
            raise RuntimeError('Thread name "%s" is not valid.' % t.name)
        if self._scan_threads and self.current_module_handle in self._curr_readout_threads:
            raise RuntimeError('Thread "%s" is already actively reading FIFO.')
        with self._readout_lock:
            self._curr_readout_threads.append(self.current_module_handle)
        self._starting_readout_event.clear()
        while not self._starting_readout_event.wait(0.01):
            if self.abort_run.is_set():
                break
            with self._readout_lock:
                if len(set(self._curr_readout_threads) & set([t.name for t in self._scan_threads if t.is_alive()])) == len(set([t.name for t in self._scan_threads if t.is_alive()])) or not self._scan_threads:
                    if not self.fifo_readout.is_running:
                        self.fifo_readout.start(fifos=self._selected_fifos, callback=callback, errback=errback, reset_rx=reset_rx, reset_fifo=reset_fifo, fill_buffer=fill_buffer, no_data_timeout=no_data_timeout, filter_func=self._filter, converter_func=self._converter, fifo_select=self._readout_fifos, enabled_fe_channels=enabled_fe_channels)
                        self._starting_readout_event.set()