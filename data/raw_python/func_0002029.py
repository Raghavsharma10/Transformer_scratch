def get(self, permission_id, expand=[]):
        """
        List a specific permisison for the given object.

        :param str permission_id: the id of the Permission to be listed.
        """
        parent_url = self.client.get_url(self.parent_object._manager._URL_KEY, 'GET', 'single', {'id': self.parent_object.id})
        target_url = parent_url + self.client.get_url_path(
            self._URL_KEY, 'GET', 'single', {'permission_id': permission_id})
        return self._get(target_url, expand=expand)