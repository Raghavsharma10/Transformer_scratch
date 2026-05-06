def _write_pidfile(self):
        """Create, write to, and lock the PID file."""
        flags = os.O_CREAT | os.O_RDWR
        try:
            # Some systems don't have os.O_EXLOCK
            flags = flags | os.O_EXLOCK
        except AttributeError:
            pass
        self._pid_fd = os.open(self.pidfile, flags, 0o666 & ~self.umask)
        os.write(self._pid_fd, str(os.getpid()).encode('utf-8'))