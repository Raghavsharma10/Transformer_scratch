def option(self, url, headers=None, kwargs=None):
        """Make a OPTION request.

        To make a OPTION request pass, ``url``

        :param url: ``str``
        :param headers: ``dict``
        :param kwargs: ``dict``
        """
        return self._request(
            method='option',
            url=url,
            headers=headers,
            kwargs=kwargs
        )