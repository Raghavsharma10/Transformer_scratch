def remove_log_type(self, logType, _assert=False):
        """
        Remove a logtype.

        :Parameters:
           #. logType (string): The logtype.
           #. _assert (boolean): Raise an assertion error if logType is not defined.
        """
        # check logType
        if _assert:
            assert logType in self.__logTypeStdoutFlags.keys(), "logType '%s' is not defined" %logType
        # remove logType
        self.__logTypeColor.pop(logType)
        self.__logTypeHighlight.pop(logType)
        self.__logTypeAttributes.pop(logType)
        self.__logTypeNames.pop(logType)
        self.__logTypeLevels.pop(logType)
        self.__logTypeFormat.pop(logType)
        self.__logTypeStdoutFlags.pop(logType)
        self.__logTypeFileFlags.pop(logType)
        self.__forcedStdoutLevels.pop(logType, None)
        self.__forcedFileLevels.pop(logType, None)