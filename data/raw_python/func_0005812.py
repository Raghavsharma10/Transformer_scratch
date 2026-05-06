def lock_file(self, fpath, after_setup=False, wait=False):
        """Locks the specified file.

        :param str|unicode fpath: File path.

        :param bool after_setup:
            True  - after logging/daemon setup
            False - before starting

        :param bool wait:
            True  - wait if locked
            False - exit if locked

        """
        command = 'flock-wait' if wait else 'flock'

        if after_setup:
            command = '%s2' % command

        self._set(command, fpath)

        return self._section