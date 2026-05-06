def web_address(self):
        """
        Return the url of the web server or None if not running
        """
        port = self._current_web_port()
        address = self.address or '127.0.0.1'
        if port is None:
            return None
        return 'http://{0}:{1}/'.format(
            address if address and not is_boot2docker() else docker_host(),
            port)