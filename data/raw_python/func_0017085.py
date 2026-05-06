def create_release(self, tag_name, target_commitish=None, name=None,
                       body=None, draft=False, prerelease=False):
        """Create a release for this repository.

        :param str tag_name: (required), name to give to the tag
        :param str target_commitish: (optional), vague concept of a target,
            either a SHA or a branch name.
        :param str name: (optional), name of the release
        :param str body: (optional), description of the release
        :param bool draft: (optional), whether this release is a draft or not
        :param bool prerelease: (optional), whether this is a prerelease or
            not
        :returns: :class:`Release <github3.repos.release.Release>`
        """
        data = {'tag_name': str(tag_name),
                'target_commitish': target_commitish,
                'name': name,
                'body': body,
                'draft': draft,
                'prerelease': prerelease
                }
        self._remove_none(data)

        url = self._build_url('releases', base_url=self._api)
        json = self._json(self._post(
            url, data=data, headers=Release.CUSTOM_HEADERS
            ), 201)
        return Release(json, self)