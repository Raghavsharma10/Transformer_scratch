def mid_generate(self):
        """Generate mid. TODO : check."""
        self.last_mid += 1
        if self.last_mid == 0:
            self.last_mid += 1
        return self.last_mid