def set_harakiri_params(self, timeout=None, verbose=None, disable_for_arh=None):
        """Sets workers harakiri parameters.

        :param int timeout: Harakiri timeout in seconds.
            Every request that will take longer than the seconds specified
            in the harakiri timeout will be dropped and the corresponding
            worker is thereafter recycled.

        :param bool verbose: Harakiri verbose mode.
            When a request is killed by Harakiri you will get a message in the uWSGI log.
            Enabling this option will print additional info (for example,
            the current syscall will be reported on Linux platforms).

        :param bool disable_for_arh: Disallow Harakiri killings during after-request hook methods.

        """
        self._set('harakiri', timeout)
        self._set('harakiri-verbose', verbose, cast=bool)
        self._set('harakiri-no-arh', disable_for_arh, cast=bool)

        return self._section