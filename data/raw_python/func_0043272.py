def detectWindowsMobile(self):
        """Return detection of Windows Mobile

        Detects if the current browser is a Windows Mobile device.
        Excludes Windows Phone 7 devices.
        Focuses on Windows Mobile 6.xx and earlier.
        """
        #Exclude new Windows Phone.
        if self.detectWindowsPhone():
            return False
        #Most devices use 'Windows CE', but some report 'iemobile'
        #  and some older ones report as 'PIE' for Pocket IE.
        #  We also look for instances of HTC and Windows for many of their WinMo devices.
        if UAgentInfo.deviceWinMob in self.__userAgent \
           or UAgentInfo.deviceIeMob in self.__userAgent \
           or UAgentInfo.enginePie in self.__userAgent:
            return True
        # Test for certain Windwos Mobile-based HTC devices.
        if UAgentInfo.manuHtc in self.__userAgent \
           and UAgentInfo.deviceWindows in self.__userAgent:
            return True
        if self.detectWapWml() \
           and UAgentInfo.deviceWindows in self.__userAgent:
            return True

        #Test for Windows Mobile PPC but not old Macintosh PowerPC.
        return UAgentInfo.devicePpc in self.__userAgent \
            and UAgentInfo.deviceMacPpc not in self.__userAgent