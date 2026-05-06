def delete_file(self, path, message, sha, branch=None, committer=None,
                    author=None):
        """Delete the file located at ``path``.

        This is part of the Contents CrUD (Create Update Delete) API. See
        http://developer.github.com/v3/repos/contents/#delete-a-file for more
        information.

        :param str path: (required), path to the file being removed
        :param str message: (required), commit message for the deletion
        :param str sha: (required), blob sha of the file being removed
        :param str branch: (optional), if not provided, uses the repository's
            default branch
        :param dict committer: (optional), if no information is given the
            authenticated user's information will be used. You must specify
            both a name and email.
        :param dict author: (optional), if omitted this will be filled in with
            committer information. If passed, you must specify both a name and
            email.
        :returns: :class:`Commit <github3.git.Commit>` if successful

        """
        json = None
        if path and message and sha:
            url = self._build_url('contents', path, base_url=self._api)
            data = {'message': message, 'sha': sha, 'branch': branch,
                    'committer': validate_commmitter(committer),
                    'author': validate_commmitter(author)}
            self._remove_none(data)
            json = self._json(self._delete(url, data=dumps(data)), 200)
            if json and 'commit' in json:
                json = Commit(json['commit'])
        return json