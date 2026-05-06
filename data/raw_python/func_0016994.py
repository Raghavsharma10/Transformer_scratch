def create_key(self, title, key):
        """Create a new key for the authenticated user.

        :param str title: (required), key title
        :param key: (required), actual key contents, accepts path as a string
            or file-like object
        :returns: :class:`Key <github3.users.Key>`
        """
        created = None

        if title and key:
            url = self._build_url('user', 'keys')
            req = self._post(url, data={'title': title, 'key': key})
            json = self._json(req, 201)
            if json:
                created = Key(json, self)
        return created