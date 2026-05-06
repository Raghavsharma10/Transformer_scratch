def detectSymbianOS(self):
        """Return detection of SymbianOS

        Detects if the current device is any Symbian OS-based device,
        including older S60, Series 70, Series 80, Series 90, and UIQ,
        or other browsers running on these devices.
        """
        return UAgentInfo.deviceSymbian in self.__userAgent \
            or UAgentInfo.deviceS60 in self.__userAgent \
            or UAgentInfo.deviceS70 in self.__userAgent \
            or UAgentInfo.deviceS80 in self.__userAgent \
            or UAgentInfo.deviceS90 in self.__userAgent