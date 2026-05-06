def _log(self, level, msg, *args, **kwargs):
        """
        Log 'msg % args' with the integer severity 'level'.

        To pass exception information, use the keyword argument exc_info with
        a true value, e.g.

        logger.log(level, "We have a %s", "mysterious problem", exc_info=1)
        """
        if not isinstance(level, int):
            if logging.raiseExceptions:
                raise TypeError('Level must be an integer!')
            else:
                return

        if self.logger.isEnabledFor(level=level):
            """
            Low-level logging routine which creates a LogRecord and then calls
            all the handlers of this logger to handle the record.
            """
            exc_info = kwargs.get('exc_info', None)
            extra = kwargs.get('extra', None)
            stack_info = kwargs.get('stack_info', False)
            record_filter = kwargs.get('record_filter', None)

            tb_info = None
            if _logone_src:
                # IronPython doesn't track Python frames, so findCaller raises an
                # exception on some versions of IronPython. We trap it here so that
                # IronPython can use logging.
                try:
                    fn, lno, func, tb_info = self.__find_caller(stack_info=stack_info)
                except ValueError:  # pragma: no cover
                    fn, lno, func = '(unknown file)', 0, '(unknown function)'
            else:  # pragma: no cover
                fn, lno, func = '(unknown file)', 0, '(unknown function)'

            if exc_info:
                if sys.version_info[0] >= 3:
                    if isinstance(exc_info, BaseException):
                        # noinspection PyUnresolvedReferences
                        exc_info = type(exc_info), exc_info, exc_info.__traceback__
                    elif not isinstance(exc_info, tuple):
                        exc_info = sys.exc_info()
                else:
                    if not isinstance(exc_info, tuple):
                        exc_info = sys.exc_info()

            if sys.version_info[0] >= 3:
                # noinspection PyArgumentList
                record = self.logger.makeRecord(self.name, level, fn, lno, msg, args,
                                                exc_info, func, extra, tb_info)
            else:
                record = self.logger.makeRecord(self.name, level, fn, lno, msg, args,
                                                exc_info, func, extra)

            if record_filter:
                record = record_filter(record)

            self.logger.handle(record=record)