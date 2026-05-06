def delete(self, url, headers=None, kwargs=None):
        """Make a DELETE request.

        To make a DELETE request pass, ``url``

        :param url: ``str``
        :param headers: ``dict``
        :param kwargs: ``dict``
        """
        return self._request(
            method='delete',
            url=url,
            headers=headers,
            kwargs=kwargs
        )