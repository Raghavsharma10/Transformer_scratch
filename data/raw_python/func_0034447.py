def info(self, callback=None, **kwargs):
        """
        Get the basic info from the current cluster.
        """
        self.client.fetch(
            self.mk_req('', method='GET', **kwargs),
            callback = callback
        )