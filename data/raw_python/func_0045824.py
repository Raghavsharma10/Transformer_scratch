def key_field_name(self):
        """
        Field identified as the key.
        """
        name = 'resource_id'
        if self.resource:
            key_field = getmeta(self.resource).key_field
            if key_field:
                name = key_field.attname
        return name