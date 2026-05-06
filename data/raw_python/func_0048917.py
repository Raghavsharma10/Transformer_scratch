def force_log(self, logType, message, data=None, tback=None, stdout=True, file=True):
        """
        Force logging a message of a certain logtype whether logtype level is allowed or not.

        :Parameters:
           #. logType (string): A defined logging type.
           #. message (string): Any message to log.
           #. tback (None, str, list): Stack traceback to print and/or write to
              log file. In general, this should be traceback.extract_stack
           #. stdout (boolean): Whether to force logging to standard output.
           #. file (boolean): Whether to force logging to file.
        """
        # log to stdout
        log = self._format_message(logType=logType, message=message, data=data, tback=tback)
        if stdout:
            self.__log_to_stdout(self.__logTypeFormat[logType][0] + log + self.__logTypeFormat[logType][1] + "\n")
            try:
                self.__stdout.flush()
            except:
                pass
            try:
                os.fsync(self.__stdout.fileno())
            except:
                pass
        if file:
            # log to file
            self.__log_to_file(log)
            self.__log_to_file("\n")
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