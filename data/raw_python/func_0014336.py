def to_json(self):
        """
        Returns the JSON representation of the API key.
        """

        result = super(ApiKey, self).to_json()
        result.update({
            'name': self.name,
            'description': self.description,
            'accessToken': self.access_token,
            'environments': [e.to_json() for e in self.environments]
        })
        return result