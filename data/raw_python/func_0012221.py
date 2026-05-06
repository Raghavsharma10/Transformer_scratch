def switch_to_next_app(self):
        """
        switches to the next app
        """
        log.debug("switching to next app...")
        cmd, url = DEVICE_URLS["switch_to_next_app"]
        self.result = self._exec(cmd, url)