def detectSonyMylo(self):
        """Return detection of a Sony Mylo device

        Detects if the current browser is a Sony Mylo device.
        """
        return UAgentInfo.manuSony in self.__userAgent \
            and (UAgentInfo.qtembedded in self.__userAgent
                or UAgentInfo.mylocom2 in self.__userAgent)