def contributors(self):
        """Property to retrieve or access the list of contributors."""
        if not self._contributors:
            self._contributors = self.get_contributors()
        return self._contributors