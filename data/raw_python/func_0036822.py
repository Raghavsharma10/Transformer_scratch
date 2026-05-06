def add(self, filename):
        """Try to add a file."""
        basename = os.path.basename(filename)
        match = self.regexp.search(basename)
        if match:
            self.by_episode[int(match.group('ep'))].add(filename)