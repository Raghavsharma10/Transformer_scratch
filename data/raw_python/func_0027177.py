def url(self, url):
        """ Set API URL endpoint

            Args:
                url: the url of the API endpoint
        """
        if url and url.endswith('/'):
            url = url[:-1]

        self._url = url