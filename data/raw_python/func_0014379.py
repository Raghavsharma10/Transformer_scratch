def to_json(self):
        """
        Returns the JSON representation of the role.
        """

        result = super(Role, self).to_json()
        result.update({
            'name': self.name,
            'description': self.description,
            'permissions': self.permissions,
            'policies': self.policies
        })
        return result