def set_reload_on_exception_params(self, do_reload=None, etype=None, evalue=None, erepr=None):
        """Sets workers reload on exceptions parameters.

        :param bool do_reload: Reload a worker when an exception is raised.

        :param str etype: Reload a worker when a specific exception type is raised.

        :param str evalue: Reload a worker when a specific exception value is raised.

        :param str erepr: Reload a worker when a specific exception type+value (language-specific) is raised.

        """
        self._set('reload-on-exception', do_reload, cast=bool)
        self._set('reload-on-exception-type', etype)
        self._set('reload-on-exception-value', evalue)
        self._set('reload-on-exception-repr', erepr)

        return self._section