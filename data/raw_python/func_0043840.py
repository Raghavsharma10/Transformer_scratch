def merge_conflicts(self):
        """The filenames of any files with merge conflicts (a list of strings)."""
        filenames = set()
        listing = self.context.capture('hg', 'resolve', '--list')
        for line in listing.splitlines():
            tokens = line.split(None, 1)
            if len(tokens) == 2:
                status, name = tokens
                if status and name and status.upper() != 'R':
                    filenames.add(name)
        return sorted(filenames)