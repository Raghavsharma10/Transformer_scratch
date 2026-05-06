def save_site(self, create=True):
        """
        Save environment settings in the directory that need to be saved
        even when creating only a new sub-site env.
        """
        self._load_sites()
        if create:
            self.sites.append(self.site_name)

        task.save_new_site(self.site_name, self.sitedir, self.target, self.port,
            self.address, self.site_url, self.passwords)