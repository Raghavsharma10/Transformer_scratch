def create(self, permission):
        """
        Create single permission for the given object.

        :param Permission permission: A single Permission object to be set.
        """
        parent_url = self.client.get_url(self.parent_object._manager._URL_KEY, 'GET', 'single', {'id': self.parent_object.id})
        target_url = parent_url + self.client.get_url_path(self._URL_KEY, 'POST', 'single')
        r = self.client.request('POST', target_url, json=permission._serialize())
        return permission._deserialize(r.json(), self)