def all(self, *args, **kwargs):
        """
        Gets all usage periods.
        """

        return self.client._get(
            self._url(),
            {},
            headers={
                'x-contentful-enable-alpha-feature': 'usage-insights'
            }
        )