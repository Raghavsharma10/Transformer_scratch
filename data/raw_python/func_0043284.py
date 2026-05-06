def detectFirefoxOSTablet(self):
        """Return detection of a Firefox OS tablet

        Detects a tablet (probably) running the Firefox OS.
        """
        if self.detectIos() \
            or self.detectAndroid() \
            or self.detectSailfish():
            return False

        if UAgentInfo.engineFirefox in self.__userAgent \
           and UAgentInfo.deviceTablet in self.__userAgent:
            return True

        return False