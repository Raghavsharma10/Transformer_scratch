def list_tags(self, image_name):
        # type: (str) -> Iterator[str]
        """ List all tags for the given image stored in the registry.

        Args:
            image_name (str):
                The name of the image to query. The image must be present on the
                registry for this call to return any values.
        Returns:
            list[str]: List of tags for that image.
        """
        tags_url = self.registry_url + '/v2/{}/tags/list'

        r = self.get(tags_url.format(image_name), auth=self.auth)
        data = r.json()

        if 'tags' in data:
            return reversed(sorted(data['tags']))

        return []