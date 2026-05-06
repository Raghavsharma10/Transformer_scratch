def detectIphoneOrIpod(self):
        """Return detection of an iPhone or iPod Touch

        Detects if the current device is an iPhone or iPod Touch.
        """
        #We repeat the searches here because some iPods may report themselves as an iPhone, which would be okay.
        return UAgentInfo.deviceIphone in self.__userAgent \
            or UAgentInfo.deviceIpod in self.__userAgent