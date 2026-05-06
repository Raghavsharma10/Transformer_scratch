def get_volume(self):
        """
        returns the current volume
        """
        log.debug("getting volumne...")
        cmd, url = DEVICE_URLS["get_volume"]
        return self._exec(cmd, url)