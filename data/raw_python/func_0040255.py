def paths_wanted(self):
        """The set of paths where we expect to find missing nodes."""
        return set(address.new(b, target='all') for b in self.missing_nodes)