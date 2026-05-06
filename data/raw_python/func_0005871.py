def add_logger_route(self, logger, matcher, requests_only=False):
        """Log to the specified named logger if regexp applied on log item matches.

        :param str|unicode|list|Logger|list[Logger] logger: Logger to associate route with.

        :param str|unicode matcher: Regular expression to apply to log item.

        :param bool requests_only: Matching should be used only for requests information messages.

        """
        command = 'log-req-route' if requests_only else 'log-route'

        for logger in listify(logger):
            self._set(command, '%s %s' % (logger, matcher), multi=True)

        return self._section