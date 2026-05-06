def detectAndroidTablet(self):
        """Return detection of an Android tablet

        Detects if the current device is a (self-reported) Android tablet.
        Google says these devices will have 'Android' and NOT 'mobile' in their user agent.
        """
        #First, let's make sure we're on an Android device.
        if not self.detectAndroid():
            return False

        #Special check for Android devices with Opera Mobile/Mini. They should NOT report here.
        if self.detectOperaMobile():
            return False

        #Otherwise, if it's Android and does NOT have 'mobile' in it, Google says it's a tablet.
        return UAgentInfo.mobile not in self.__userAgent