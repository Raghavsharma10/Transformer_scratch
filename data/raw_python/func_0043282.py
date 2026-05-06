def detectMeegoPhone(self):
        """Return detection of a Meego phone

        Detects a phone running the Meego OS.
        """
        return UAgentInfo.deviceMeego in self.__userAgent \
            and UAgentInfo.mobi in self.__userAgent