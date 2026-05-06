def put_content(self, content):
        """
        Makes a ``PUT`` request with the content in the body.

        :raise: An :exc:`requests.RequestException` if it is not 2xx.
        """

        r = requests.request(self.method if self.method else 'PUT', self.url, data=content, **self.storage_args)
        if self.raise_for_status: r.raise_for_status()