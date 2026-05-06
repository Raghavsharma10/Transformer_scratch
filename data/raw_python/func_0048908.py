def force_log_type_stdout_flag(self, logType, flag):
        """
        Force a logtype standard output logging flag despite minimum and maximum logging level boundaries.

        :Parameters:
           #. logType (string): A defined logging type.
           #. flag (None boolean): The standard output logging flag.
              If None, logtype existing forced flag is released.
        """
        assert logType in self.__logTypeStdoutFlags.keys(), "logType '%s' not defined" %logType
        if flag is None:
            self.__forcedStdoutLevels.pop(logType, None)
            self.__update_stdout_flags()
        else:
            assert isinstance(flag, bool), "flag must be boolean"
            self.__logTypeStdoutFlags[logType] = flag
            self.__forcedStdoutLevels[logType] = flag