def detectBlackBerry(self):
        """Return detection of Blackberry

        Detects if the current browser is any BlackBerry.
        Includes the PlayBook.
        """
        return UAgentInfo.deviceBB in self.__userAgent \
            or UAgentInfo.vndRIM in self.__httpAccept