def to_json(self):
        """
        Returns the JSON Representation of the UI extension.
        """

        result = super(UIExtension, self).to_json()
        result.update({
            'extension': self.extension
        })

        return result