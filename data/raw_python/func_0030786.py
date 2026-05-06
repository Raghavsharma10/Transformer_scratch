def promote(self):
        """ Mark object as alive, so it won't be collected during next
        run of the garbage collector.
        """
        if self.expiry is not None:
            self.promoted = self.time_module.time() + self.expiry