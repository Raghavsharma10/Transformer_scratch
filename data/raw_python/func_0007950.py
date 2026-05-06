def get(self,
            resource_id=None,
            resource_action=None,
            resource_cls=None,
            single_resource=False):
        """ Gets the details for one or more resources by ID

        Args:
            cls - gophish.models.Model - The resource class
            resource_id - str - The endpoint (URL path) for the resource
            resource_action - str - An action to perform on the resource
            resource_cls - cls - A class to use for parsing, if different than
                the base resource
            single_resource - bool - An override to tell Gophish that even
                though we aren't requesting a single resource, we expect a
                single response object

        Returns:
            One or more instances of cls parsed from the returned JSON
        """

        endpoint = self.endpoint

        if not resource_cls:
            resource_cls = self._cls

        if resource_id:
            endpoint = self._build_url(endpoint, resource_id)

        if resource_action:
            endpoint = self._build_url(endpoint, resource_action)

        response = self.api.execute("GET", endpoint)
        if not response.ok:
            raise Error.parse(response.json())

        if resource_id or single_resource:
            return resource_cls.parse(response.json())

        return [resource_cls.parse(resource) for resource in response.json()]