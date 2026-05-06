def detectWebOSTV(self):
        """Return detection of a WebOS smart TV

        Detects if the current browser is on a WebOS smart TV.
        """
        return UAgentInfo.deviceWebOStv in self.__userAgent \
            and UAgentInfo.smartTV2 in self.__userAgent