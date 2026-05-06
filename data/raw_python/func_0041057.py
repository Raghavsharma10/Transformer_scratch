def crossref_paths(self):
        """Just like crossrefs, but all the targets are munged to :all."""
        return set(
            [address.new(repo=x.repo, path=x.path) for x in self.crossrefs])