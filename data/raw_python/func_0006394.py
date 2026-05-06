def exit_sync(self):
        ''' Waiting for all threads to appear, then continue.
        '''
        if self._scan_threads and self.current_module_handle not in [t.name for t in self._scan_threads]:
            raise RuntimeError('Thread name "%s" is not valid.')
        if self._scan_threads and self.current_module_handle not in self._curr_sync_threads:
            raise RuntimeError('Thread "%s" is not reading FIFO.')
        with self._sync_lock:
            self._curr_sync_threads.remove(self.current_module_handle)
        self._exit_sync_event.clear()
        while not self._exit_sync_event.wait(0.01):
            if self.abort_run.is_set():
                break
            with self._sync_lock:
                if len(set(self._curr_sync_threads) & set([t.name for t in self._scan_threads if t.is_alive()])) == 0 or not self._scan_threads:
                    self._exit_sync_event.set()