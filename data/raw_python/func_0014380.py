def to_json(self):
        """
        Returns the JSON representation of the space membership.
        """

        result = super(SpaceMembership, self).to_json()
        result.update({
            'admin': self.admin,
            'roles': self.roles
        })
        return result