def create(self, resource_id=None, attributes=None, **kwargs):
        """
        Creates an entry with a given ID (optional) and attributes.
        """

        if self.content_type_id is not None:
            if attributes is None:
                attributes = {}
            attributes['content_type_id'] = self.content_type_id

        return super(EntriesProxy, self).create(resource_id=resource_id, attributes=attributes)