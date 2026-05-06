def detectBlackBerryTouch(self):
        """Return detection of a Blackberry touchscreen device

        Detects if the current browser is a BlackBerry Touch
        device, such as the Storm, Torch, and Bold Touch. Excludes the Playbook.
        """
        return UAgentInfo.deviceBBStorm in self.__userAgent \
                or UAgentInfo.deviceBBTorch in self.__userAgent \
                or UAgentInfo.deviceBBBoldTouch in self.__userAgent \
                or UAgentInfo.deviceBBCurveTouch in self.__userAgent