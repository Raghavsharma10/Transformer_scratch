def gist(self, id_num):
        """Gets the gist using the specified id number.

        :param int id_num: (required), unique id of the gist
        :returns: :class:`Gist <github3.gists.Gist>`
        """
        url = self._build_url('gists', str(id_num))
        json = self._json(self._get(url), 200)
        return Gist(json, self) if json else None