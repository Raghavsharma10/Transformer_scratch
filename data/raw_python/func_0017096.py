def key(self, id_num):
        """Get the specified deploy key.

        :param int id_num: (required), id of the key
        :returns: :class:`Key <github3.users.Key>` if successful, else None
        """
        json = None
        if int(id_num) > 0:
            url = self._build_url('keys', str(id_num), base_url=self._api)
            json = self._json(self._get(url), 200)
        return Key(json, self) if json else None