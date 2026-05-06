def to_json(self):
        """
        Returns the JSON representation of the environment.
        """

        result = super(Environment, self).to_json()
        result.update({
            'name': self.name
        })

        return result