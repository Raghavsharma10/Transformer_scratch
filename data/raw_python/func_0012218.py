def get_wifi_state(self):
        """
        returns the current Wi-Fi state the device is connected to
        """
        log.debug("getting wifi state...")
        cmd, url = DEVICE_URLS["get_wifi_state"]
        return self._exec(cmd, url)