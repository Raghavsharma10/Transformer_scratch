def fetch_resource(self):
        """
        Fetch & return the resource that the action operated on, or `None` if
        the resource no longer exists (specifically, if the API returns a 404)

        :rtype: `Droplet`, `Image`, `FloatingIP`, or `None`
        :raises ValueError: if the action has an unknown ``resource_type``
            (This indicates a deficiency in the library; please report it!)
        :raises DOAPIError: if the API endpoint replies with a non-404 error
        """
        try:
            if self.resource_type == "droplet":
                return self.doapi_manager.fetch_droplet(self.resource_id)
            elif self.resource_type == "image":
                return self.doapi_manager.fetch_image(self.resource_id)
            elif self.resource_type == "floating_ip":
                return self.doapi_manager.fetch_floating_ip(self.resource_id)
            else:
                raise ValueError('{0.resource_type!r}: unknown resource_type'\
                                 .format(self))
        except DOAPIError as e:
            if e.response.status_code == 404:
                return None
            else:
                raise