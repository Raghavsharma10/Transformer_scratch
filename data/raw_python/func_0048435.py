def put(self, url, headers=None, body=None, kwargs=None):
        """Make a PUT request.

        To make a PUT request pass, ``url``

        :param url: ``str``
        :param headers: ``dict``
        :param body: ``object``
        :param kwargs: ``dict``
        """
        return self._request(
            method='put',
            url=url,
            headers=headers,
            body=body,
            kwargs=kwargs
        )