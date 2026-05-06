def set_filters(self, include=None, exclude=None, write_errors=None, write_errors_tolerance=None, sigpipe=None):
        """Set various log data filters.

        :param str|unicode|list include: Show only log lines matching the specified regexp.

            .. note:: Requires enabled PCRE support.

        :param str|unicode|list exclude: Do not show log lines matching the specified regexp.

            .. note:: Requires enabled PCRE support.

        :param bool write_errors: Log (annoying) write()/writev() errors. Default: ``True``.

            .. note:: If both this and ``sigpipe`` set to ``False``, it's the same
               as setting ``write-errors-exception-only`` uWSGI option.

        :param int write_errors_tolerance: Set the maximum number of allowed write errors before exception
            is raised. Default: no tolerance.

            .. note:: Available for Python, Perl, PHP.

        :param bool sigpipe: Log (annoying) SIGPIPE. Default: ``True``.

            .. note:: If both this and ``write_errors`` set to ``False``, it's the same
               as setting ``write-errors-exception-only`` uWSGI option.

        """
        if write_errors is not None:
            self._set('ignore-write-errors', not write_errors, cast=bool)

        if sigpipe is not None:
            self._set('ignore-sigpipe', not sigpipe, cast=bool)

        self._set('write-errors-tolerance', write_errors_tolerance)

        for line in listify(include):
            self._set('log-filter', line, multi=True)

        for line in listify(exclude):
            self._set('log-drain', line, multi=True)

        return self._section