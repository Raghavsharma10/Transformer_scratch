def detectUbuntuPhone(self):
        """Return detection of an Ubuntu Mobile OS phone

        Detects a phone running the Ubuntu Mobile OS.
        """
        if UAgentInfo.deviceUbuntu in self.__userAgent \
           and UAgentInfo.mobile in self.__userAgent:
            return True

        return False