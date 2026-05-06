def detectTierRichCss(self):
        """Return detection of any device in the 'Rich CSS' Tier

        The quick way to detect for a tier of devices.
        This method detects for devices which are likely to be capable
        of viewing CSS content optimized for the iPhone,
        but may not necessarily support JavaScript.
        Excludes all iPhone Tier devices.
        """
        #The following devices are explicitly ok.
        #Note: 'High' BlackBerry devices ONLY
        if not self.detectMobileQuick():
            return False
        #Exclude iPhone Tier and e-Ink Kindle devices
        if self.detectTierIphone() \
            or self.detectKindle():
            return False
        #The following devices are explicitly ok.
        #Note: 'High' BlackBerry devices ONLY
        #Older Windows 'Mobile' isn't good enough for iPhone Tier.
        return self.detectWebkit() \
            or self.detectS60OssBrowser() \
            or self.detectBlackBerryHigh() \
            or self.detectWindowsMobile() \
            or UAgentInfo.engineTelecaQ in self.__userAgent