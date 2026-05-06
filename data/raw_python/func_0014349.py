def to_json(self):
        """
        Returns the JSON representation of the space.
        """

        result = super(Space, self).to_json()
        result.update({'name': self.name})
        return result