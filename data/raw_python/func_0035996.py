def start(self):
        """Start the daemon."""
        if self.worker is None:
            raise DaemonError('No worker is defined for daemon')

        if os.environ.get('DAEMONOCLE_RELOAD'):
            # If this is actually a reload, we need to wait for the
            # existing daemon to exit first
            self._emit_message('Reloading {prog} ... '.format(prog=self.prog))
            # Orhpan this process so the parent can exit
            self._orphan_this_process(wait_for_parent=True)
            pid = self._read_pidfile()
            if (pid is not None and
                    self._pid_is_alive(pid, timeout=self.stop_timeout)):
                # The process didn't exit for some reason
                self._emit_failed()
                message = ('Previous process (PID {pid}) did NOT '
                           'exit during reload').format(pid=pid)
                self._emit_error(message)
                self._shutdown(message, 1)

        # Check to see if the daemon is already running
        pid = self._read_pidfile()
        if pid is not None:
            # I don't think this should not be a fatal error
            self._emit_warning('{prog} already running with PID {pid}'.format(
                prog=self.prog, pid=pid))
            return

        if not self.detach and not os.environ.get('DAEMONOCLE_RELOAD'):
            # This keeps the original parent process open so that we
            # maintain control of the tty
            self._fork_and_supervise_child()

        if not os.environ.get('DAEMONOCLE_RELOAD'):
            # A custom message is printed for reloading
            self._emit_message('Starting {prog} ... '.format(prog=self.prog))

        self._setup_environment()

        if self.detach:
            self._detach_process()
        else:
            self._emit_ok()

        if self.pidfile is not None:
            self._write_pidfile()

        # Setup signal handlers
        signal.signal(signal.SIGINT, self._handle_terminate)
        signal.signal(signal.SIGQUIT, self._handle_terminate)
        signal.signal(signal.SIGTERM, self._handle_terminate)

        self._run()