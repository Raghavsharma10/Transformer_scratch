def detectBlackBerry10Phone(self):
        """Return detection of a Blackberry 10 OS phone

        Detects if the current browser is a BlackBerry 10 OS phone.
        Excludes the PlayBook.
        """
        return UAgentInfo.deviceBB10 in self.__userAgent \
            and UAgentInfo.mobile in self.__userAgent