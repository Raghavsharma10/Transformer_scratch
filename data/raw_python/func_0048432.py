def post(self, url, headers=None, body=None, kwargs=None):
        """Make a POST request.

        To make a POST request pass, ``url``

        :param url: ``str``
        :param headers: ``dict``
        :param body: ``object``
        :param kwargs: ``dict``
        """
        return self._request(
            method='post',
            url=url,
            headers=headers,
            body=body,
            kwargs=kwargs
        )