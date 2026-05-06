def detectUbuntuTablet(self):
        """Return detection of an Ubuntu Mobile OS tablet

        Detects a tablet running the Ubuntu Mobile OS.
        """
        if UAgentInfo.deviceUbuntu in self.__userAgent \
           and UAgentInfo.deviceTablet in self.__userAgent:
            return True

        return False