def _lock(self):
        """Lock the config DB."""
        if not self.locked:
            self.device.cu.lock()
            self.locked = True