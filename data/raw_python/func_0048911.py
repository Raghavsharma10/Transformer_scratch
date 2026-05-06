def set_log_type_name(self, logType, name):
        """
        Set a logtype name.

        :Parameters:
           #. logType (string): A defined logging type.
           #. name (string): The logtype new name.
        """
        assert logType in self.__logTypeStdoutFlags.keys(), "logType '%s' not defined" %logType
        assert isinstance(name, basestring), "name must be a string"
        name = str(name)
        self.__logTypeNames[logType] = name