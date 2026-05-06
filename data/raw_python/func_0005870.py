def add_logger(self, logger, requests_only=False, for_single_worker=False):
        """Set/add a common logger or a request requests only.

        :param str|unicode|list|Logger|list[Logger] logger:

        :param bool requests_only: Logger used only for requests information messages.

        :param bool for_single_worker: Logger to be used in single-worker setup.


        """
        if for_single_worker:
            command = 'worker-logger-req' if requests_only else 'worker-logger'
        else:
            command = 'req-logger' if requests_only else 'logger'

        for logger in listify(logger):
            self._set(command, logger, multi=True)

        return self._section