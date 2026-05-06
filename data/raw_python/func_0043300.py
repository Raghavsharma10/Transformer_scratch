def detectTierIphone(self):
        """Return detection of any device in the iPhone/Android/WP7/WebOS Tier

        The quick way to detect for a tier of devices.
        This method detects for devices which can
        display iPhone-optimized web content.
        Includes iPhone, iPod Touch, Android, Windows Phone 7, Palm WebOS, etc.
        """
        return self.__isIphone \
            or self.__isAndroidPhone \
            or self.detectWindowsPhone() \
            or self.detectBlackBerry10Phone() \
            or self.detectPalmWebOS() \
            or self.detectBada() \
            or self.detectTizen() \
            or self.detectFirefoxOSPhone() \
            or self.detectSailfishPhone() \
            or self.detectUbuntuPhone() \
            or self.detectGamingHandheld()