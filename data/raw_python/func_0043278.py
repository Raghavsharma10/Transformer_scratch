def detectWebOSTablet(self):
        """Return detection of an HP WebOS tablet

        Detects if the current browser is on an HP tablet running WebOS.
        """
        return UAgentInfo.deviceWebOShp in self.__userAgent \
            and UAgentInfo.deviceTablet in self.__userAgent