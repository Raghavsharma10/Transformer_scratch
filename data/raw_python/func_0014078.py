def _unlock(self):
        """Unlock the config DB."""
        if self.locked:
            self.device.cu.unlock()
            self.locked = False