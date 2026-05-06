def _url(self, url, file_upload=False):
        """
        Creates the request URL.
        """

        host = self.api_url
        if file_upload:
            host = self.uploads_api_url

        protocol = 'https' if self.https else 'http'
        if url.endswith('/'):
            url = url[:-1]
        return '{0}://{1}/{2}'.format(
            protocol,
            host,
            url
        )