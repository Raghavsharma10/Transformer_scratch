def run_run(self, run, conf=None, run_conf=None, use_thread=False, catch_exception=True):
        '''Runs a run in another thread. Non-blocking.

        Parameters
        ----------
        run : class, object
            Run class or object.
        run_conf : str, dict, file
            Specific configuration for the run.
        use_thread : bool
            If True, run run in thread and returns blocking function.

        Returns
        -------
        If use_thread is True, returns function, which blocks until thread terminates, and which itself returns run status.
        If use_thread is False, returns run status.
        '''
        if isinstance(conf, basestring) and os.path.isfile(conf):
            logging.info('Updating configuration from file %s', os.path.abspath(conf))
        elif conf is not None:
            logging.info('Updating configuration')
        conf = self.open_conf(conf)
        self._conf.update(conf)

        if isclass(run):
            # instantiate the class
            run = run(conf=self._conf)

        local_run_conf = {}
        # general parameters from conf
        if 'run_conf' in self._conf:
            logging.info('Updating run configuration using run_conf key from configuration')
            local_run_conf.update(self._conf['run_conf'])
        # check for class name, scan specific parameters from conf
        if run.__class__.__name__ in self._conf:
            logging.info('Updating run configuration using %s key from configuration' % (run.__class__.__name__,))
            local_run_conf.update(self._conf[run.__class__.__name__])

        if isinstance(run_conf, basestring) and os.path.isfile(run_conf):
            logging.info('Updating run configuration from file %s', os.path.abspath(run_conf))
        elif run_conf is not None:
            logging.info('Updating run configuration')
        run_conf = self.open_conf(run_conf)
        # check for class name, scan specific parameters from conf
        if run.__class__.__name__ in run_conf:
            run_conf = run_conf[run.__class__.__name__]
        # run_conf parameter has highest priority, updated last
        local_run_conf.update(run_conf)

        if use_thread:
            self.current_run = run

            @thunkify(thread_name='RunThread', daemon=True, default_func=self.current_run.get_run_status)
            def run_run_in_thread():
                return run.run(run_conf=local_run_conf)

            signal.signal(signal.SIGINT, self._signal_handler)
            logging.info('Press Ctrl-C to stop run')

            return run_run_in_thread()
        else:
            self.current_run = run
            status = run.run(run_conf=local_run_conf)
            if not catch_exception and status != run_status.finished:
                raise RuntimeError('Exception occurred. Please read the log.')
            return status