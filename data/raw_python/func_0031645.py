def fetch_watch_info(self):
        """
        This method should be called before accessing :attr:`watch_info`, :attr:`firmware_version`
        or :attr:`watch_platform`. Blocks until it has fetched the required information.
        """
        self._watch_info = self.send_and_read(WatchVersion(data=WatchVersionRequest()), WatchVersion).data