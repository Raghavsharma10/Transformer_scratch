def set_log_file_extension(self, logFileExtension):
        """
        Set the log file extension.

        :Parameters:
           #. logFileExtension (string): Logging file extension. A logging file full name is
              set as logFileBasename.logFileExtension
        """
        assert isinstance(logFileExtension, basestring), "logFileExtension must be a basestring"
        assert len(logFileExtension), "logFileExtension can't be empty"
        if logFileExtension[0] == ".":
            logFileExtension = logFileExtension[1:]
        assert len(logFileExtension), "logFileExtension is not allowed to be single dot"
        if logFileExtension[-1] == ".":
            logFileExtension = logFileExtension[:-1]
        assert len(logFileExtension), "logFileExtension is not allowed to be double dots"
        self.__logFileExtension = logFileExtension
        # set log file name
        self.__set_log_file_name()