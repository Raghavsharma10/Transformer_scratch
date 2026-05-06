def load_from_stream(self, dim):
        """Load from an NCStream object."""
        self.unlimited = dim.isUnlimited
        self.private = dim.isPrivate
        self.vlen = dim.isVlen
        if not self.vlen:
            self.size = dim.length