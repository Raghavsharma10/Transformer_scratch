def detectWapWml(self):
        """Return detection of a WAP- or WML-capable device

        Detects whether the device supports WAP or WML.
        """
        return UAgentInfo.vndwap in self.__httpAccept \
            or UAgentInfo.wml in self.__httpAccept