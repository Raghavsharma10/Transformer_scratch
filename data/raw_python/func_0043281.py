def detectTizenTV(self):
        """Return detection of a Tizen smart TV

        Detects if the current browser is on a Tizen smart TV.
        """
        return UAgentInfo.deviceTizen in self.__userAgent \
            and UAgentInfo.smartTV1 in self.__userAgent