def detectMaemoTablet(self):
        """Return detection of a Maemo OS tablet

        Detects if the current device is on one of the Maemo-based Nokia Internet Tablets.
        """
        if UAgentInfo.maemo in self.__userAgent:
            return True

        return UAgentInfo.linux in self.__userAgent \
            and UAgentInfo.deviceTablet in self.__userAgent \
            and not self.detectWebOSTablet() \
            and not self.detectAndroid()