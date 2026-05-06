def add_collaborator(self, login):
        """Add ``login`` as a collaborator to a repository.

        :param str login: (required), login of the user
        :returns: bool -- True if successful, False otherwise
        """
        resp = False
        if login:
            url = self._build_url('collaborators', login, base_url=self._api)
            resp = self._boolean(self._put(url), 204, 404)
        return resp