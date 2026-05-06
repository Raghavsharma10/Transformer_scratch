def tick(self, d):
        """ ticks come in on every block
        """
        if self.test_blocks:
            if not (self.counter["blocks"] or 0) % self.test_blocks:
                self.test()
            self.counter["blocks"] += 1