def head(self, url, headers=None, kwargs=None):
        """Make a HEAD request.

        To make a HEAD request pass, ``url``

        :param url: ``str``
        :param headers: ``dict``
        :param kwargs: ``dict``
        """
        return self._request(
            method='head',
            url=url,
            headers=headers,
            kwargs=kwargs
        )