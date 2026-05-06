def is_touched(self, pin):
        """Return True if the specified pin is being touched, otherwise returns
        False.
        """
        assert pin >= 0 and pin < 12, 'pin must be between 0-11 (inclusive)'
        t = self.touched()
        return (t & (1 << pin)) > 0