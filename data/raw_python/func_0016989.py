def authorization(self, id_num):
        """Get information about authorization ``id``.

        :param int id_num: (required), unique id of the authorization
        :returns: :class:`Authorization <Authorization>`
        """
        json = None
        if int(id_num) > 0:
            url = self._build_url('authorizations', str(id_num))
            json = self._json(self._get(url), 200)
        return Authorization(json, self) if json else None