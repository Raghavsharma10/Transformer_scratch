def create_team(self, name, repo_names=[], permission=''):
        """Assuming the authenticated user owns this organization,
        create and return a new team.

        :param str name: (required), name to be given to the team
        :param list repo_names: (optional) repositories, e.g.
            ['github/dotfiles']
        :param str permission: (optional), options:

            - ``pull`` -- (default) members can not push or administer
                repositories accessible by this team
            - ``push`` -- members can push and pull but not administer
                repositories accessible by this team
            - ``admin`` -- members can push, pull and administer
                repositories accessible by this team

        :returns: :class:`Team <Team>`
        """
        data = {'name': name, 'repo_names': repo_names,
                'permission': permission}
        url = self._build_url('teams', base_url=self._api)
        json = self._json(self._post(url, data), 201)
        return Team(json, self._session) if json else None