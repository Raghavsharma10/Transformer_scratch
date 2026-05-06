def resolve(self, space_id=None, environment_id=None):
        """
        Resolves link to a specific resource.
        """

        proxy_method = getattr(
            self._client,
            base_path_for(self.link_type)
        )
        if self.link_type == 'Space':
            return proxy_method().find(self.id)
        elif environment_id is not None:
            return proxy_method(space_id, environment_id).find(self.id)
        else:
            return proxy_method(space_id).find(self.id)