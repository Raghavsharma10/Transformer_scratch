def post(self, resource):
        """ Creates a new instance of the resource.

        Args:
            resource - gophish.models.Model - The resource instance

        """
        response = self.api.execute(
            "POST", self.endpoint, json=(resource.as_dict()))

        if not response.ok:
            raise Error.parse(response.json())

        return self._cls.parse(response.json())