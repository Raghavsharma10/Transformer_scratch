def detectOperaMobile(self):
        """Return detection of an Opera browser for a mobile device

        Detects Opera Mobile or Opera Mini.
        """
        return UAgentInfo.engineOpera in self.__userAgent \
            and (UAgentInfo.mini in self.__userAgent
                or UAgentInfo.mobi in self.__userAgent)