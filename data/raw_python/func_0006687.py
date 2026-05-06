def set_level(self, level):
        """
        Set the logging level of this logger.

        :param level: must be an int or a str.
        """
        for handler in self.__coloredlogs_handlers:
            handler.setLevel(level=level)

        self.logger.setLevel(level=level)