def _read_pidfile(self):
        """Read the PID file and check to make sure it's not stale."""
        if self.pidfile is None:
            return None

        if not os.path.isfile(self.pidfile):
            return None

        # Read the PID file
        with open(self.pidfile, 'r') as fp:
            try:
                pid = int(fp.read())
            except ValueError:
                self._emit_warning('Empty or broken pidfile {pidfile}; '
                                   'removing'.format(pidfile=self.pidfile))
                pid = None

        if pid is not None and psutil.pid_exists(pid):
            return pid
        else:
            # Remove the stale PID file
            os.remove(self.pidfile)
            return None