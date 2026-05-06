def detectSmartphone(self):
        """Return detection of a general smartphone device

        Checks to see whether the device is *any* 'smartphone'.
        Note: It's better to use DetectTierIphone() for modern touchscreen devices.
        """
        return self.detectTierIphone() \
            or self.detectS60OssBrowser() \
            or self.detectSymbianOS() \
            or self.detectWindowsMobile() \
            or self.detectBlackBerry() \
            or self.detectMeegoPhone() \
            or self.detectPalmWebOS()