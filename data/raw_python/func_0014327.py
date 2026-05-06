def to_json(self):
        """
        Returns the JSON Representation of the resource.
        """

        result = super(FieldsResource, self).to_json()
        result['fields'] = self.fields_with_locales()
        return result