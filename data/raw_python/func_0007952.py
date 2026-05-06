def put(self, resource):
        """ Edits an existing resource

        Args:
            resource - gophish.models.Model - The resource instance
        """

        endpoint = self.endpoint

        if resource.id:
            endpoint = self._build_url(endpoint, resource.id)

        response = self.api.execute("PUT", endpoint, json=resource.as_dict())

        if not response.ok:
            raise Error.parse(response.json())

        return self._cls.parse(response.json())