def detectFirefoxOSPhone(self):
        """Return detection of a Firefox OS phone

        Detects a phone (probably) running the Firefox OS.
        """
        if self.detectIos() \
            or self.detectAndroid() \
            or self.detectSailfish():
            return False

        if UAgentInfo.engineFirefox in self.__userAgent \
           and UAgentInfo.mobile in self.__userAgent:
            return True

        return False