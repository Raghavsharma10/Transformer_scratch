def set_log_type_level(self, logType, level):
        """
        Set a logtype logging level.

        :Parameters:
           #. logType (string): A defined logging type.
           #. level (number): The level of logging.
        """
        assert _is_number(level), "level must be a number"
        level = float(level)
        name = str(name)
        self.__logTypeLevels[logType] = level