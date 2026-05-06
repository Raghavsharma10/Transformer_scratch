def get_display(self):
        """
        returns information about the display, including
        brightness, screensaver etc.
        """
        log.debug("getting display information...")
        cmd, url = DEVICE_URLS["get_display"]
        return self._exec(cmd, url)