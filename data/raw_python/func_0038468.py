def _get_url(cls, administration_id: int, resource_path: str):
        """
        Builds the URL to the API endpoint specified by the given parameters.

        :param administration_id: The ID of the administration (may be None).
        :param resource_path: The path to the resource.
        :return: The absolute URL to the endpoint.
        """
        url = urljoin(cls.base_url, '%s/' % cls.version)

        if administration_id is not None:
            url = urljoin(url, '%s/' % administration_id)

        url = urljoin(url, '%s.json' % resource_path)

        return url