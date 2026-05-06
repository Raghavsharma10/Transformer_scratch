def HeadList(self):
        """Return a list of all the currently loaded repo HEAD objects."""
        return [(rname, repo.currenthead) for rname, repo in self.repos.items()
                ]