def get_bluetooth_state(self):
        """
        returns the bluetooth state
        """
        log.debug("getting bluetooth state...")
        cmd, url = DEVICE_URLS["get_bluetooth_state"]
        return self._exec(cmd, url)