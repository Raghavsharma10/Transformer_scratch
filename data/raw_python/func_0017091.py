def edit(self,
             name,
             description=None,
             homepage=None,
             private=None,
             has_issues=None,
             has_wiki=None,
             has_downloads=None,
             default_branch=None):
        """Edit this repository.

        :param str name: (required), name of the repository
        :param str description: (optional), If not ``None``, change the
            description for this repository. API default: ``None`` - leave
            value unchanged.
        :param str homepage: (optional), If not ``None``, change the homepage
            for this repository. API default: ``None`` - leave value unchanged.
        :param bool private: (optional), If ``True``, make the repository
            private. If ``False``, make the repository public. API default:
            ``None`` - leave value unchanged.
        :param bool has_issues: (optional), If ``True``, enable issues for
            this repository. If ``False``, disable issues for this repository.
            API default: ``None`` - leave value unchanged.
        :param bool has_wiki: (optional), If ``True``, enable the wiki for
            this repository. If ``False``, disable the wiki for this
            repository. API default: ``None`` - leave value unchanged.
        :param bool has_downloads: (optional), If ``True``, enable downloads
            for this repository. If ``False``, disable downloads for this
            repository. API default: ``None`` - leave value unchanged.
        :param str default_branch: (optional), If not ``None``, change the
            default branch for this repository. API default: ``None`` - leave
            value unchanged.
        :returns: bool -- True if successful, False otherwise
        """
        edit = {'name': name, 'description': description, 'homepage': homepage,
                'private': private, 'has_issues': has_issues,
                'has_wiki': has_wiki, 'has_downloads': has_downloads,
                'default_branch': default_branch}
        self._remove_none(edit)
        json = None
        if edit:
            json = self._json(self._patch(self._api, data=dumps(edit)), 200)
            self._update_(json)
            return True
        return False