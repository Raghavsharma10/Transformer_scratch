def create(self, data):
        """Create a new component
        """
        response = self.http.post(str(self), json=data, auth=self.auth)
        response.raise_for_status()
        return response.json()