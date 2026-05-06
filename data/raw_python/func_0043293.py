def detectMidpCapable(self):
        """Return detection of a MIDP mobile Java-capable device

        Detects if the current device supports MIDP, a mobile Java technology.
        """
        return UAgentInfo.deviceMidp in self.__userAgent \
            or UAgentInfo.deviceMidp in self.__httpAccept