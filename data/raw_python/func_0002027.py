def set(self, permissions):
        """
        Set the object permissions. If the parent object already has permissions, they will be overwritten.

        :param [] permissions: A group of Permission objects to be set.
        """
        parent_url = self.client.get_url(self.parent_object._manager._URL_KEY, 'GET', 'single', {'id': self.parent_object.id})
        target_url = parent_url + self.client.get_url_path(self._URL_KEY, 'PUT', 'multi')
        r = self.client.request('PUT', target_url, json=permissions)
        if r.status_code != 201:
            raise exceptions.ServerError("Expected 201 response, got %s: %s" % (r.status_code, target_url))
        return self.list()