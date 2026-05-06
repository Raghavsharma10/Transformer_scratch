def send_signal(self, s):
        """
        Send a signal to the daemon process.

        The signal must have been enabled using the ``signals``
        parameter of :py:meth:`Service.__init__`. Otherwise, a
        ``ValueError`` is raised.
        """
        self._get_signal_event(s)  # Check if signal has been enabled
        pid = self.get_pid()
        if not pid:
            raise ValueError('Daemon is not running.')
        os.kill(pid, s)