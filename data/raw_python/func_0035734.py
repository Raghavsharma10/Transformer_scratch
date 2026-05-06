def devices(self):
        """
        Return a list of connected devices in the form (*serial*, *status*) where status can
        be any of the following:

        1. device
        2. offline
        3. unauthorized

        :returns: A list of tuples representing connected devices
        """
        devices = None
        with self.socket.Connect():
            devices = self._command("host:devices")

        return parse_device_list(devices)