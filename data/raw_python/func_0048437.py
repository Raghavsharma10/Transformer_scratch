def get(self, url, headers=None, kwargs=None):
        """Make a GET request.

        To make a GET request pass, ``url``

        :param url: ``str``
        :param headers: ``dict``
        :param kwargs: ``dict``
        """
        return self._request(
            method='get',
            url=url,
            headers=headers,
            kwargs=kwargs
        )