def create_key(self, title, key):
        """Create a deploy key.

        :param str title: (required), title of key
        :param str key: (required), key text
        :returns: :class:`Key <github3.users.Key>` if successful, else None
        """
        json = None
        if title and key:
            data = {'title': title, 'key': key}
            url = self._build_url('keys', base_url=self._api)
            json = self._json(self._post(url, data=data), 201)
        return Key(json, self) if json else None