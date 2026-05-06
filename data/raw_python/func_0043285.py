def detectSailfishPhone(self):
        """Return detection of a Sailfish phone

        Detects a phone running the Sailfish OS.
        """
        if self.detectSailfish() \
           and UAgentInfo.mobile in self.__userAgent:
            return True

        return False