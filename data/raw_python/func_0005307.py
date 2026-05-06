def list_images(self):
        # type: () -> List[str]
        """ List images stored in the registry.

        Returns:
            list[str]: List of image names.
        """
        r = self.get(self.registry_url + '/v2/_catalog', auth=self.auth)
        return r.json()['repositories']