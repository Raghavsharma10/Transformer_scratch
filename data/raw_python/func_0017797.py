def _load_sites(self):
        """
        Gets the names of all of the sites from the datadir and stores them
        in self.sites. Also returns this list.
        """
        if not self.sites:
            self.sites = task.list_sites(self.datadir)
        return self.sites