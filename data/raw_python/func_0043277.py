def detectPalmOS(self):
        """Return detection of a PalmOS device

        Detects if the current browser is on a PalmOS device.
        """
        #Most devices nowadays report as 'Palm', but some older ones reported as Blazer or Xiino.
        if UAgentInfo.devicePalm in self.__userAgent \
           or UAgentInfo.engineBlazer in self.__userAgent \
           or UAgentInfo.engineXiino in self.__userAgent:
            # Make sure it's not WebOS
            return not self.detectPalmWebOS()
        return False