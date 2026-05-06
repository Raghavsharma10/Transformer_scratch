def add_file(self, f):
        """Add a partition identity as a child of a dataset identity."""

        if not self.files:
            self.files = set()

        self.files.add(f)

        self.locations.set(f.type_)