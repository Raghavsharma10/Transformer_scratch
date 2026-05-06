def detectNintendo(self):
        """Return detection of Nintendo

        Detects if the current device is a Nintendo game device.
        """
        return UAgentInfo.deviceNintendo in self.__userAgent \
            or UAgentInfo.deviceNintendo in self.__userAgent \
            or UAgentInfo.deviceNintendo in self.__userAgent