def detectDangerHiptop(self):
        """Return detection of a Danger Hiptop

        Detects the Danger Hiptop device.
        """
        return UAgentInfo.deviceDanger in self.__userAgent \
            or UAgentInfo.deviceHiptop in self.__userAgent