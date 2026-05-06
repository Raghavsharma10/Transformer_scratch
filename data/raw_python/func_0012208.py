def get_notifications(self):
        """
        returns the list of all notifications in queue
        """
        log.debug("getting notifications in queue...")
        cmd, url = DEVICE_URLS["get_notifications_queue"]
        return self._exec(cmd, url)