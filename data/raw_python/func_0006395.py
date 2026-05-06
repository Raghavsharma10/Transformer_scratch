def readout(self, *args, **kwargs):
        ''' Running the FIFO readout while executing other statements.

        Starting and stopping of the FIFO readout is synchronized between the threads.
        '''
        timeout = kwargs.pop('timeout', 10.0)
        self.start_readout(*args, **kwargs)
        try:
            yield
        finally:
            try:
                self.stop_readout(timeout=timeout)
            except Exception:
                # in case something fails, call this on last resort
                # if run was aborted, immediately stop readout
                if self.abort_run.is_set():
                    with self._readout_lock:
                        if self.fifo_readout.is_running:
                            self.fifo_readout.stop(timeout=0.0)