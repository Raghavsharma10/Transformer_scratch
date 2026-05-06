def set_autostep(self, val):
        """Set autostep value (property)"""
        if val is None:
            # disabled by default for pdsh compat (+inf is 1E400, but a bug in
            # python 2.4 makes it impossible to be pickled, so we use less)
            # NOTE: Later, we could consider sys.maxint here
            self._autostep = 1E100
        else:
            # - 1 because user means node count, but we means real steps
            self._autostep = int(val) - 1