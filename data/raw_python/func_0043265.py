def detectIphone(self):
        """Return detection of an iPhone

        Detects if the current device is an iPhone.
        """
        # The iPad and iPod touch say they're an iPhone! So let's disambiguate.
        return UAgentInfo.deviceIphone in self.__userAgent \
            and not self.detectIpad() \
            and not self.detectIpod()