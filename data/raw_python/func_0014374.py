def update(self, attributes=None):
        """
        Updates the entry with attributes.
        """

        if attributes is None:
            attributes = {}

        attributes['content_type_id'] = self.sys['content_type'].id

        return super(Entry, self).update(attributes)