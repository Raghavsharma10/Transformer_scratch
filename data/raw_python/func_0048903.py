def set_log_file(self, logfile):
        """
        Set the log file full path including directory path basename and extension.

        :Parameters:
           #. logFile (string): the full log file path including basename and
              extension. If this is given, all of logFileBasename and logFileExtension
              will be discarded. logfile is equivalent to logFileBasename.logFileExtension
        """
        assert isinstance(logfile, basestring), "logfile must be a string"
        basename, extension = os.path.splitext(logfile)
        self.__set_log_file_basename(logFileBasename=basename)
        self.set_log_file_extension(logFileExtension=extension)