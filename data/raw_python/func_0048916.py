def log(self, logType, message, data=None, tback=None):
        """
        log a message of a certain logtype.

        :Parameters:
           #. logType (string): A defined logging type.
           #. message (string): Any message to log.
           #. data (None,  object): Any type of data to print and/or write to log file
              after log message
           #. tback (None, str, list): Stack traceback to print and/or write to
              log file. In general, this should be traceback.extract_stack
        """
        # log to stdout
        log = self._format_message(logType=logType, message=message, data=data, tback=tback)
        if self.__logToStdout and self.__logTypeStdoutFlags[logType]:
            self.__log_to_stdout(self.__logTypeFormat[logType][0] + log + self.__logTypeFormat[logType][1] + "\n")
            if self.__flush:
                try:
                    self.__stdout.flush()
                except:
                    pass
                try:
                    os.fsync(self.__stdout.fileno())
                except:
                    pass
        # log to file
        if self.__logToFile and self.__logTypeFileFlags[logType]:
            self.__log_to_file(log)
            self.__log_to_file("\n")
            if self.__flush:
                try:
                    self.__logFileStream.flush()
                except:
                    pass
                try:
                    os.fsync(self.__logFileStream.fileno())
                except:
                    pass
        # set last logged message
        self.__lastLogged[logType] = log
        self.__lastLogged[-1]      = log