def set_log_type_flags(self, logType, stdoutFlag, fileFlag):
        """
        Set a defined log type flags.

        :Parameters:
           #. logType (string): A defined logging type.
           #. stdoutFlag (boolean): Whether to log to the standard output stream.
           #. fileFlag (boolean): Whether to log to to file.
        """
        assert logType in self.__logTypeStdoutFlags.keys(), "logType '%s' not defined" %logType
        assert isinstance(stdoutFlag, bool), "stdoutFlag must be boolean"
        assert isinstance(fileFlag, bool), "fileFlag must be boolean"
        self.__logTypeStdoutFlags[logType] = stdoutFlag
        self.__logTypeFileFlags[logType]   = fileFlag