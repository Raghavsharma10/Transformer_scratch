def get_gist(self):
        """Retrieve the gist at this version.

        :returns: :class:`Gist <github3.gists.gist.Gist>`

        """
        from .gist import Gist
        json = self._json(self._get(self._api), 200)
        return Gist(json, self)