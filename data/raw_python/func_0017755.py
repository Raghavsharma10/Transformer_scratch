def stop(self, block=False):
        """
        Tell the daemon process to stop.

        Sends the SIGTERM signal to the daemon process, requesting it
        to terminate.

        If ``block`` is true then the call blocks until the daemon
        process has exited. This may take some time since the daemon
        process will complete its on-going backup activities before
        shutting down. ``block`` can either be ``True`` (in which case
        it blocks indefinitely) or a timeout in seconds.

        The return value is ``True`` if the daemon process has been
        stopped and ``False`` otherwise.

        .. versionadded:: 0.3
            The ``block`` parameter
        """
        self.send_signal(signal.SIGTERM)
        return _block(lambda: not self.is_running(), block)