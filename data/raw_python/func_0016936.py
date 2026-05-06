def create_repo(self,
                    name,
                    description='',
                    homepage='',
                    private=False,
                    has_issues=True,
                    has_wiki=True,
                    has_downloads=True,
                    team_id=0,
                    auto_init=False,
                    gitignore_template=''):
        """Create a repository for this organization if the authenticated user
        is a member.

        :param str name: (required), name of the repository
        :param str description: (optional)
        :param str homepage: (optional)
        :param bool private: (optional), If ``True``, create a private
            repository. API default: ``False``
        :param bool has_issues: (optional), If ``True``, enable issues for
            this repository. API default: ``True``
        :param bool has_wiki: (optional), If ``True``, enable the wiki for
            this repository. API default: ``True``
        :param bool has_downloads: (optional), If ``True``, enable downloads
            for this repository. API default: ``True``
        :param int team_id: (optional), id of the team that will be granted
            access to this repository
        :param bool auto_init: (optional), auto initialize the repository.
        :param str gitignore_template: (optional), name of the template; this
            is ignored if auto_int = False.
        :returns: :class:`Repository <github3.repos.Repository>`

        .. warning: ``name`` should be no longer than 100 characters
        """
        url = self._build_url('repos', base_url=self._api)
        data = {'name': name, 'description': description,
                'homepage': homepage, 'private': private,
                'has_issues': has_issues, 'has_wiki': has_wiki,
                'has_downloads': has_downloads, 'auto_init': auto_init,
                'gitignore_template': gitignore_template}
        if team_id > 0:
            data.update({'team_id': team_id})
        json = self._json(self._post(url, data), 201)
        return Repository(json, self) if json else None