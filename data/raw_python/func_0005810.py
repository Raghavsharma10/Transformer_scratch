def set_basic_params(self, count=None, thunder_lock=None, lock_engine=None):
        """
        :param int count: Create the specified number of shared locks.

        :param bool thunder_lock: Serialize accept() usage (if possible)
            Could improve performance on Linux with robust pthread mutexes.

            http://uwsgi.readthedocs.io/en/latest/articles/SerializingAccept.html

        :param str|unicode lock_engine: Set the lock engine.

            Example:
                - ipcsem

        """
        self._set('thunder-lock', thunder_lock, cast=bool)
        self._set('lock-engine', lock_engine)
        self._set('locks', count)

        return self._section