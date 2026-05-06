def edit(self, tag_name=None, target_commitish=None, name=None, body=None,
             draft=None, prerelease=None):
        """Users with push access to the repository can edit a release.

        If the edit is successful, this object will update itself.

        :param str tag_name: (optional), Name of the tag to use
        :param str target_commitish: (optional), The "commitish" value that
            determines where the Git tag is created from. Defaults to the
            repository's default branch.
        :param str name: (optional), Name of the release
        :param str body: (optional), Description of the release
        :param boolean draft: (optional), True => Release is a draft
        :param boolean prerelease: (optional), True => Release is a prerelease
        :returns: True if successful; False if not successful
        """
        url = self._api
        data = {
            'tag_name': tag_name,
            'target_commitish': target_commitish,
            'name': name,
            'body': body,
            'draft': draft,
            'prerelease': prerelease,
        }
        self._remove_none(data)

        r = self._session.patch(
            url, data=json.dumps(data), headers=Release.CUSTOM_HEADERS
        )

        successful = self._boolean(r, 200, 404)
        if successful:
            # If the edit was successful, let's update the object.
            self.__init__(r.json(), self)

        return successful