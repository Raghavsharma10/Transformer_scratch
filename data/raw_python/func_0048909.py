def force_log_type_file_flag(self, logType, flag):
        """
        Force a logtype file logging flag despite minimum and maximum logging level boundaries.

        :Parameters:
           #. logType (string): A defined logging type.
           #. flag (None, boolean): The file logging flag.
              If None, logtype existing forced flag is released.
        """
        assert logType in self.__logTypeStdoutFlags.keys(), "logType '%s' not defined" %logType
        if flag is None:
            self.__forcedFileLevels.pop(logType, None)
            self.__update_file_flags()
        else:
            assert isinstance(flag, bool), "flag must be boolean"
            self.__logTypeFileFlags[logType] = flag
            self.__forcedFileLevels[logType] = flag