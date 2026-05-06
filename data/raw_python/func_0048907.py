def set_minimum_level(self, level=0, stdoutFlag=True, fileFlag=True):
        """
        Set the minimum logging level. All levels below the minimum will be ignored at logging.

        :Parameters:
           #. level (None, number, str): The minimum level of logging.
              If None, minimum level checking is left out.
              If str, it must be a defined logtype and therefore the minimum level would be the level of this logtype.
           #. stdoutFlag (boolean): Whether to apply this minimum level to standard output logging.
           #. fileFlag (boolean): Whether to apply this minimum level to file logging.
        """
        # check flags
        assert isinstance(stdoutFlag, bool), "stdoutFlag must be boolean"
        assert isinstance(fileFlag, bool), "fileFlag must be boolean"
        if not (stdoutFlag or fileFlag):
            return
        # check level
        if level is not None:
            if isinstance(level, basestring):
                level = str(level)
                assert level in self.__logTypeStdoutFlags.keys(), "level '%s' given as string, is not defined logType" %level
                level = self.__logTypeLevels[level]
            assert _is_number(level), "level must be a number"
            level = float(level)
            if stdoutFlag:
                if self.__stdoutMaxLevel is not None:
                    assert level<=self.__stdoutMaxLevel, "stdoutMinLevel must be smaller or equal to stdoutMaxLevel %s"%self.__stdoutMaxLevel
            if fileFlag:
                if self.__fileMaxLevel is not None:
                    assert level<=self.__fileMaxLevel, "fileMinLevel must be smaller or equal to fileMaxLevel %s"%self.__fileMaxLevel
        # set flags
        if stdoutFlag:
            self.__stdoutMinLevel = level
            self.__update_stdout_flags()
        if fileFlag:
            self.__fileMinLevel = level
            self.__update_file_flags()