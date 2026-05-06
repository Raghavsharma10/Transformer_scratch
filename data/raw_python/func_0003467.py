def start(self):
        """Start the background process."""
        self._lc = LoopingCall(self._download)
        # Run immediately, and then every 30 seconds:
        self._lc.start(30, now=True)