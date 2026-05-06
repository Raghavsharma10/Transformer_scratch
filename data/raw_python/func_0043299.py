def detectTierTablet(self):
        """Return detection of any device in the Tablet Tier

        The quick way to detect for a tier of devices.
        This method detects for the new generation of
        HTML 5 capable, larger screen tablets.
        Includes iPad, Android (e.g., Xoom), BB Playbook, WebOS, etc.
        """
        return self.detectIpad() \
            or self.detectAndroidTablet() \
            or self.detectBlackBerryTablet() \
            or self.detectFirefoxOSTablet() \
            or self.detectUbuntuTablet() \
            or self.detectWebOSTablet()