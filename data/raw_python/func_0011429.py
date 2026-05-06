def _pfp__snapshot(self, recurse=True):
        """Save off the current value of the field
        """
        super(Array, self)._pfp__snapshot(recurse=recurse)
        self.snapshot_raw_data = self.raw_data

        if recurse:
            for item in self.items:
                item._pfp__snapshot(recurse=recurse)