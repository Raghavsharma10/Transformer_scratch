def add_logger_encoder(self, encoder, logger=None, requests_only=False, for_single_worker=False):
        """Add an item in the log encoder or request encoder chain.

        * http://uwsgi-docs.readthedocs.io/en/latest/LogEncoders.html

            .. note:: Encoders automatically enable master log handling (see ``.set_master_logging_params()``).

            .. note:: For best performance consider allocating a thread
                for log sending with ``dedicate_thread``.

        :param str|unicode|list|Encoder encoder: Encoder (or a list) to add into processing.

        :param str|unicode|Logger logger: Logger apply associate encoders to.

        :param bool requests_only: Encoder to be used only for requests information messages.

        :param bool for_single_worker: Encoder to be used in single-worker setup.

        """
        if for_single_worker:
            command = 'worker-log-req-encoder' if requests_only else 'worker-log-encoder'
        else:
            command = 'log-req-encoder' if requests_only else 'log-encoder'

        for encoder in listify(encoder):

            value = '%s' % encoder

            if logger:
                if isinstance(logger, Logger):
                    logger = logger.alias

                value += ':%s' % logger

            self._set(command, value, multi=True)

        return self._section