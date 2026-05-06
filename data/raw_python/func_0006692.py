def use_loggly(self, enabled=True,
                   loggly_token=None,
                   loggly_tag=None,
                   level=logging.WARNING,
                   log_format=None,
                   date_format=None):
        """
        Enable handler for sending the record to Loggly service.
        """
        if enabled:
            if not self.__loggly_handler:
                assert loggly_token, 'Loggly token is missing!'

                # Use logger name for default Loggly tag
                if not loggly_tag:
                    loggly_tag = self.name

                # Create new LogglyHandler instance
                self.__loggly_handler = LogglyHandler(token=loggly_token, tag=loggly_tag)

                # Use this format for default case
                if not log_format:
                    log_format = '{"name":"%(name)s","process":"%(process)d",' \
                                 '"levelname":"%(levelname)s","time":"%(asctime)s",' \
                                 '"filename":"%(filename)s","programname":"%(programname)s",' \
                                 '"module":"%(module)s","funcName":"%(funcName)s",' \
                                 '"lineno":"%(lineno)d","message":"%(message)s"}'

                # Set formatter
                formatter = logging.Formatter(fmt=log_format, datefmt=date_format)
                self.__loggly_handler.setFormatter(fmt=formatter)

                # Set level for this handler
                self.__loggly_handler.setLevel(level=level)

                # Add this handler to logger
                self.add_handler(hdlr=self.__loggly_handler)
        elif self.__loggly_handler:
            # Remove handler from logger
            self.remove_handler(hdlr=self.__loggly_handler)
            self.__loggly_handler = None