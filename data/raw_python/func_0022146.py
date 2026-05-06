def view_rest_retry(self, url=None):
        """
        View current rest retry settings in the `requests.Session()` object

        **Parameters:**

          - **url:** URL to use to determine retry methods for. Defaults to 'https://'

        **Returns:** Dict, Key header, value is header value.
        """
        if url is None:
            url = "https://"
        return vars(self._session.get_adapter(url).max_retries)