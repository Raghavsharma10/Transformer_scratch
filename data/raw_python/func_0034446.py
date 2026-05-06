def ping(self, callback=None, **kwargs):
        """
        Ping request to check status of elasticsearch host
        """
        self.client.fetch(
            self.mk_req('', method='HEAD', **kwargs),
            callback = callback
        )