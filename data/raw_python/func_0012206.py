def get_device_state(self):
        """
        returns the full device state
        """
        log.debug("getting device state...")
        cmd, url = DEVICE_URLS["get_device_state"]
        return self._exec(cmd, url)