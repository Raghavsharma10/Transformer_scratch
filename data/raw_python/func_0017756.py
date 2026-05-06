def kill(self, block=False):
        """
        Kill the daemon process.

        Sends the SIGKILL signal to the daemon process, killing it. You
        probably want to try :py:meth:`stop` first.

        If ``block`` is true then the call blocks until the daemon
        process has exited. ``block`` can either be ``True`` (in which
        case it blocks indefinitely) or a timeout in seconds.

        Returns ``True`` if the daemon process has (already) exited and
        ``False`` otherwise.

        The PID file is always removed, whether the process has already
        exited or not. Note that this means that subsequent calls to
        :py:meth:`is_running` and :py:meth:`get_pid` will behave as if
        the process has exited. If you need to be sure that the process
        has already exited, set ``block`` to ``True``.

        .. versionadded:: 0.5.1
            The ``block`` parameter
        """
        pid = self.get_pid()
        if not pid:
            raise ValueError('Daemon is not running.')
        try:
            os.kill(pid, signal.SIGKILL)
            return _block(lambda: not self.is_running(), block)
        except OSError as e:
            if e.errno == errno.ESRCH:
                raise ValueError('Daemon is not running.')
            raise
        finally:
            self.pid_file.release()