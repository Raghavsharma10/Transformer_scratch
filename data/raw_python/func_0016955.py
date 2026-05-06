def delete(self, message, branch=None, committer=None, author=None):
        """Delete this file.

        :param str message: (required), commit message to describe the removal
        :param str branch: (optional), branch where the file exists.
            Defaults to the default branch of the repository.
        :param dict committer: (optional), if no information is given the
            authenticated user's information will be used. You must specify
            both a name and email.
        :param dict author: (optional), if omitted this will be filled in with
            committer information. If passed, you must specify both a name and
            email.
        :returns: :class:`Commit <github3.git.Commit>`

        """
        json = None
        if message:
            data = {'message': message, 'sha': self.sha, 'branch': branch,
                    'committer': validate_commmitter(committer),
                    'author': validate_commmitter(author)}
            self._remove_none(data)
            json = self._json(self._delete(self._api, data=dumps(data)), 200)
            if 'commit' in json:
                json = Commit(json['commit'], self)
        return json