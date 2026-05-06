def list(self):
        """
        List permissions for the given object.
        """
        parent_url = self.client.get_url(self.parent_object._manager._URL_KEY, 'GET', 'single', {'id': self.parent_object.id})
        target_url = parent_url + self.client.get_url_path(self._URL_KEY, 'GET', 'multi')
        return base.Query(self, target_url)