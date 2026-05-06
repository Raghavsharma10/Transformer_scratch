def detectAndroidPhone(self):
        """Return  detection of an Android phone

        Detects if the current device is a (small-ish) Android OS-based device
        used for calling and/or multi-media (like a Samsung Galaxy Player).
        Google says these devices will have 'Android' AND 'mobile' in user agent.
        Ignores tablets (Honeycomb and later).
        """
        #First, let's make sure we're on an Android device.
        if not self.detectAndroid():
            return False

        #If it's Android and has 'mobile' in it, Google says it's a phone.
        if UAgentInfo.mobile in self.__userAgent:
            return True

        #Special check for Android devices with Opera Mobile/Mini. They should report here.
        if self.detectOperaMobile():
            return True

        return False