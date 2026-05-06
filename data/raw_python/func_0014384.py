def to_json(self):
        """
        Returns the JSON Representation of the content type field.
        """

        result = {
            'name': self.name,
            'id': self._real_id(),
            'type': self.type,
            'localized': self.localized,
            'omitted': self.omitted,
            'required': self.required,
            'disabled': self.disabled,
            'validations': [v.to_json() for v in self.validations]
        }

        if self.type == 'Array':
            result['items'] = self.items

        if self.type == 'Link':
            result['linkType'] = self.link_type

        return result