def detectTizen(self):
        """Return detection of a Tizen device

        Detects a device running the Tizen smartphone OS.
        """
        return UAgentInfo.deviceTizen in self.__userAgent \
            and UAgentInfo.mobile in self.__userAgent