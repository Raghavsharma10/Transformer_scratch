def create(self, attributes=None, **kwargs):
        """
        Creates a space with given attributes.
        """

        if attributes is None:
            attributes = {}
        if 'default_locale' not in attributes:
            attributes['default_locale'] = self.client.default_locale

        return super(SpacesProxy, self).create(resource_id=None, attributes=attributes)