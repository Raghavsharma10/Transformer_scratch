def write(self, value):
        """Write a (new) value to this variable."""
        assert self.num_write_waits > 0, self
        self.num_write_waits -= 1
        self.values.append(value)
        if self.readable:
            LOG.debug('%s is now readable', self.name)