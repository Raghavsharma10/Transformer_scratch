def to_json(self):
        """
        Returns the JSON representation of the content type.
        """

        result = super(ContentType, self).to_json()
        result.update({
            'name': self.name,
            'description': self.description,
            'displayField': self.display_field,
            'fields': [f.to_json() for f in self.fields]
        })
        return result