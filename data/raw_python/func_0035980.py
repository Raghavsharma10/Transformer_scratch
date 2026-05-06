def _setup_piddir(self):
        """Create the directory for the PID file if necessary."""
        if self.pidfile is None:
            return
        piddir = os.path.dirname(self.pidfile)
        if not os.path.isdir(piddir):
            # Create the directory with sensible mode and ownership
            os.makedirs(piddir, 0o777 & ~self.umask)
            os.chown(piddir, self.uid, self.gid)