def default_logger(self, name=__name__, enable_stream=False,
                       enable_file=True):
        """Default Logger.

        This is set to use a rotating File handler and a stream handler.
        If you use this logger all logged output that is INFO and above will
        be logged, unless debug_logging is set then everything is logged.
        The logger will send the same data to a stdout as it does to the
        specified log file.

        You can disable the default handlers by setting either `enable_file` or
        `enable_stream` to `False`

        :param name: ``str``
        :param enable_stream: ``bol``
        :param enable_file: ``bol``
        :return: ``object``
        """
        if self.format is None:
            self.format = logging.Formatter(
                '%(asctime)s - %(module)s:%(levelname)s => %(message)s'
            )

        log = logging.getLogger(name)
        self.name = name

        if enable_file is True:
            file_handler = handlers.RotatingFileHandler(
                filename=self.return_logfile(filename='%s.log' % name),
                maxBytes=self.max_size,
                backupCount=self.max_backup
            )
            self.set_handler(log, handler=file_handler)

        if enable_stream is True or self.debug_logging is True:
            stream_handler = logging.StreamHandler()
            self.set_handler(log, handler=stream_handler)

        log.info('Logger [ %s ] loaded', name)
        return log