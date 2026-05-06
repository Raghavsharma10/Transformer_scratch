def use_file(self, enabled=True,
                 file_name=None,
                 level=logging.WARNING,
                 when='d',
                 interval=1,
                 backup_count=30,
                 delay=False,
                 utc=False,
                 at_time=None,
                 log_format=None,
                 date_format=None):
        """
        Handler for logging to a file, rotating the log file at certain timed intervals.
        """
        if enabled:
            if not self.__file_handler:
                assert file_name, 'File name is missing!'

                # Create new TimedRotatingFileHandler instance
                kwargs = {
                    'filename': file_name,
                    'when': when,
                    'interval': interval,
                    'backupCount': backup_count,
                    'encoding': 'UTF-8',
                    'delay': delay,
                    'utc': utc,
                }

                if sys.version_info[0] >= 3:
                    kwargs['atTime'] = at_time

                self.__file_handler = TimedRotatingFileHandler(**kwargs)

                # Use this format for default case
                if not log_format:
                    log_format = '%(asctime)s %(name)s[%(process)d] ' \
                                 '%(programname)s/%(module)s/%(funcName)s[%(lineno)d] ' \
                                 '%(levelname)s %(message)s'

                # Set formatter
                formatter = logging.Formatter(fmt=log_format, datefmt=date_format)
                self.__file_handler.setFormatter(fmt=formatter)

                # Set level for this handler
                self.__file_handler.setLevel(level=level)

                # Add this handler to logger
                self.add_handler(hdlr=self.__file_handler)
        elif self.__file_handler:
            # Remove handler from logger
            self.remove_handler(hdlr=self.__file_handler)
            self.__file_handler = None