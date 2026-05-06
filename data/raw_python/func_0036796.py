def clouds(self, id=None, name=None, provider=None, search=None):
        """
        Property-like function to call the _list_clouds function in
        order to populate self._clouds dict

        :returns: A list of Cloud instances.
        """
        if self._clouds is None:
            self._clouds = {}
            self._list_clouds()

        if id:
            return [self._clouds[cloud_id] for cloud_id in self._clouds.keys()
                    if id == self._clouds[cloud_id].id]
        elif name:
            return [self._clouds[cloud_id] for cloud_id in self._clouds.keys()
                    if name == self._clouds[cloud_id].title]
        elif provider:
            return [self._clouds[cloud_id] for cloud_id in self._clouds.keys()
                    if provider == self._clouds[cloud_id].provider]
        elif search:
            return [self._clouds[cloud_id] for cloud_id in self._clouds.keys()
                    if search in self._clouds[cloud_id].title
                    or search in self._clouds[cloud_id].id
                    or search in self._clouds[cloud_id].provider]
        else:
            return [self._clouds[cloud_id] for cloud_id in self._clouds.keys()]