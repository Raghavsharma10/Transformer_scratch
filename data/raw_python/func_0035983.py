def _close_pidfile(self):
        """Closes and removes the PID file."""
        if self._pid_fd is not None:
            os.close(self._pid_fd)
        try:
            os.remove(self.pidfile)
        except OSError as ex:
            if ex.errno != errno.ENOENT:
                raise