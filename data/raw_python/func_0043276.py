def detectBlackBerryHigh(self):
        """Return detection of a Blackberry device with a better browser

        Detects if the current browser is a BlackBerry device AND
        has a more capable recent browser. Excludes the Playbook.
        Examples, Storm, Bold, Tour, Curve2
        Excludes the new BlackBerry OS 6 and 7 browser!!
        """
        #Disambiguate for BlackBerry OS 6 or 7 (WebKit) browser
        if self.detectBlackBerryWebKit():
            return False
        if not self.detectBlackBerry():
            return False

        return self.detectBlackBerryTouch() \
            or UAgentInfo.deviceBBBold in self.__userAgent \
            or UAgentInfo.deviceBBTour in self.__userAgent \
            or UAgentInfo.deviceBBCurve in self.__userAgent