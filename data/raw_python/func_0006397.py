def stop_readout(self, timeout=10.0):
        ''' Stopping the FIFO readout.

        Stopping of the FIFO readout is executed only once by a random thread.
        Stopping of the FIFO readout is synchronized between all threads reading out the FIFO.
        '''
        if self._scan_threads and self.current_module_handle not in [t.name for t in self._scan_threads]:
            raise RuntimeError('Thread name "%s" is not valid.')
        if self._scan_threads and self.current_module_handle not in self._curr_readout_threads:
            raise RuntimeError('Thread "%s" is not reading FIFO.')
        with self._readout_lock:
            self._curr_readout_threads.remove(self.current_module_handle)
        self._stopping_readout_event.clear()
        while not self._stopping_readout_event.wait(0.01):
            with self._readout_lock:
                if len(set(self._curr_readout_threads) & set([t.name for t in self._scan_threads if t.is_alive()])) == 0 or not self._scan_threads or self.abort_run.is_set():
                    if self.fifo_readout.is_running:
                        self.fifo_readout.stop(timeout=timeout)
                    self._stopping_readout_event.set()