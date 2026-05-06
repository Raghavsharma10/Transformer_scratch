def set_naming_params(self, autonaming=None, prefix=None, suffix=None, name=None):
        """Setups processes naming parameters.

        :param bool autonaming: Automatically set process name to something meaningful.
            Generated process names may be 'uWSGI Master', 'uWSGI Worker #', etc.

        :param str|unicode prefix: Add prefix to process names.

        :param str|unicode suffix: Append string to process names.

        :param str|unicode name: Set process names to given static value.

        """
        self._set('auto-procname', autonaming, cast=bool)
        self._set('procname-prefix%s' % ('-spaced' if prefix and prefix.endswith(' ') else ''), prefix)
        self._set('procname-append', suffix)
        self._set('procname', name)

        return self._section