def _get_metadata(self, data_type, key, timeout=5):
        """Get host instance metadata (only works on GCP hosts).

        More details about instance metadata:
        https://cloud.google.com/compute/docs/storing-retrieving-metadata

        Args:
            data_type (str): Type of metadata to fetch. Eg. project,
                instance
            key (str): Key of metadata to fetch
            timeout (int, optional): HTTP request timeout in seconds.
                Default is 5 seconds.
        Returns:
            (str): Plain text value of metadata entry
        Raises:
            GoogleCloudError: when request to metadata endpoint fails
        """
        endpoint_url = self.METADATA_ENDPOINT.format(
            data_type=data_type, key=key)
        try:
            rsp = requests.get(
                endpoint_url,
                headers={'Metadata-Flavor': 'Google'},
                timeout=timeout)
            rsp.raise_for_status()
        except requests.exceptions.RequestException as e:
            raise exceptions.GoogleCloudError(
                'Could not fetch "{key}" from "{type}" metadata using "{url}".'
                'Error: {e}'.format(
                    key=key, type=data_type, url=endpoint_url, e=e))
        metadata_value = rsp.text
        if metadata_value.strip() == '':
            raise exceptions.GoogleCloudError(
                'Error when fetching metadata from "{url}": server returned '
                'an empty value.'.format(url=endpoint_url))
        return metadata_value