def release_branches(self):
        """A dictionary that maps branch names to :class:`Release` objects."""
        self.ensure_release_scheme('branches')
        return dict((r.revision.branch, r) for r in self.releases.values())