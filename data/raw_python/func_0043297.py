def detectMobileQuick(self):
        """Return detection of any mobile device using the quicker method

        Detects if the current device is a mobile device.
        This method catches most of the popular modern devices.
        Excludes Apple iPads and other modern tablets.
        """
        #Let's exclude tablets
        if self.__isTierTablet:
            return False

        #Most mobile browsing is done on smartphones
        if self.detectSmartphone():
            return True

        #Catch-all for many mobile devices
        if UAgentInfo.mobile in self.__userAgent:
            return True

        if self.detectOperaMobile():
            return True

        #We also look for Kindle devices
        if self.detectKindle() \
            or self.detectAmazonSilk():
            return True

        if self.detectWapWml() \
           or self.detectMidpCapable() \
           or self.detectBrewDevice():
            return True

        if UAgentInfo.engineNetfront in self.__userAgent \
           or UAgentInfo.engineUpBrowser in self.__userAgent:
            return True

        return False