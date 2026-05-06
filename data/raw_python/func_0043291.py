def detectGamingHandheld(self):
        """Return detection of a gaming handheld with a modern iPhone-class browser

        Detects if the current device is a handheld gaming device with
        a touchscreen and modern iPhone-class browser. Includes the Playstation Vita.
        """
        return UAgentInfo.devicePlaystation in self.__userAgent \
            and UAgentInfo.devicePlaystationVita in self.__userAgent