def set_pid_file(self, fpath, before_priv_drop=True, safe=False):
        """Creates pidfile before or after privileges drop.

        :param str|unicode fpath: File path.

        :param bool before_priv_drop: Whether to create pidfile before privileges are dropped.

            .. note:: Vacuum is made after privileges drop, so it may not be able
                to delete PID file if it was created before dropping.

        :param bool safe: The safe-pidfile works similar to pidfile
            but performs the write a little later in the loading process.
            This avoids overwriting the value when app loading fails,
            with the consequent loss of a valid PID number.

        """
        command = 'pidfile'

        if not before_priv_drop:
            command += '2'

        if safe:
            command = 'safe-' + command

        self._set(command, fpath)

        return self._section