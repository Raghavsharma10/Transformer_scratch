def _init(self, run_conf, run_number=None):
        '''Initialization before a new run.
        '''
        self.stop_run.clear()
        self.abort_run.clear()
        self._run_status = run_status.running
        self._write_run_number(run_number)
        self._init_run_conf(run_conf)