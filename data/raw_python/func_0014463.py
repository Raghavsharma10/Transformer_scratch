def close(self):
        """
        Close this session, and any open channels that are tied to it.
        """
        if not self.active:
            return
        self.active = False
        self.packetizer.close()
        self.join()
        for chan in self._channels.values():
            chan._unlink()