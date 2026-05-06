def set_idle_params(self, timeout=None, exit=None):
        """Activate idle mode - put uWSGI in cheap mode after inactivity timeout.

        :param int timeout: Inactivity timeout in seconds.

        :param bool exit: Shutdown uWSGI when idle.

        """
        self._set('idle', timeout)
        self._set('die-on-idle', exit, cast=bool)

        return self._section