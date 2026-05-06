def detectAndroid(self):
        """Return detection of an Android device

        Detects *any* Android OS-based device: phone, tablet, and multi-media player.
        Also detects Google TV.
        """
        if UAgentInfo.deviceAndroid in self.__userAgent \
           or self.detectGoogleTV():
            return True

        return False