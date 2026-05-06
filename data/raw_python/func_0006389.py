def handle_err(self, exc):
        '''Handling of Exceptions.

        Parameters
        ----------
        exc : list, tuple
            Information of the exception of the format (type, value, traceback).
            Uses the return value of sys.exc_info().
        '''
        if self.reset_rx_on_error and isinstance(exc[1], (RxSyncError, EightbTenbError)):
            self.fifo_readout.print_readout_status()
            self.fifo_readout.reset_rx()
        else:
            # print just the first error massage
            if not self.abort_run.is_set():
                self.abort(msg=exc[1].__class__.__name__ + ": " + str(exc[1]))
            self.err_queue.put(exc)