def set_exception_handling_params(self, handler=None, catch=None, no_write_exception=None):
        """Exception handling related params.

        :param str|unicode|list[str|unicode] handler: Register one or more exception handling C-functions.

        :param bool catch: Catch exceptions and report them as http output (including stack trace and env params).

            .. warning:: Use only for testing purposes.

        :param bool no_write_exception: Disable exception generation on write()/writev().

            .. note:: This can be combined with ``logging.set_filters(write_errors=False, sigpipe=False)``.

            .. note: Currently available for Python.

        """
        self._set('exception-handler', handler, multi=True)
        self._set('catch-exceptions', catch, cast=bool)
        self._set('disable-write-exception', no_write_exception, cast=bool)

        return self._section