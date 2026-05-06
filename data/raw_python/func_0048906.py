def set_log_file_maximum_size(self, logFileMaxSize):
        """
        Set the log file maximum size in megabytes

        :Parameters:
           #. logFileMaxSize (number): The maximum size in Megabytes of a logging file.
              Once exceeded, another logging file as logFileBasename_N.logFileExtension
              will be created. Where N is an automatically incremented number.
        """
        assert _is_number(logFileMaxSize), "logFileMaxSize must be a number"
        logFileMaxSize = float(logFileMaxSize)
        assert logFileMaxSize>=1, "logFileMaxSize minimum size is 1 megabytes"
        self.__maxlogFileSize = logFileMaxSize