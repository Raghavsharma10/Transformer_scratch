def get_current_notification(self):
        """
        returns the current notification (i.e. the one that is visible)
        """
        log.debug("getting visible notification...")
        cmd, url = DEVICE_URLS["get_current_notification"]
        return self._exec(cmd, url)