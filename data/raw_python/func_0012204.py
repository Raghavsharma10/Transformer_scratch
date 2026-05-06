def get_endpoint_map(self):
        """
        returns API version and endpoint map
        """
        log.debug("getting end points...")
        cmd, url = DEVICE_URLS["get_endpoint_map"]
        return self._exec(cmd, url)