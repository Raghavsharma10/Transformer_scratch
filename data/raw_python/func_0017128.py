def remove_collaborator(self, login):
        """Remove collaborator ``login`` from the repository.

        :param str login: (required), login name of the collaborator
        :returns: bool
        """
        resp = False
        if login:
            url = self._build_url('collaborators', login, base_url=self._api)
            resp = self._boolean(self._delete(url), 204, 404)
        return resp