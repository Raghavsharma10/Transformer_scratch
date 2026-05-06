def patch(self, url, headers=None, body=None, kwargs=None):
        """Make a PATCH request.

        To make a PATCH request pass, ``url``

        :param url: ``str``
        :param headers: ``dict``
        :param body: ``object``
        :param kwargs: ``dict``
        """
        return self._request(
            method='patch',
            url=url,
            headers=headers,
            body=body,
            kwargs=kwargs
        )