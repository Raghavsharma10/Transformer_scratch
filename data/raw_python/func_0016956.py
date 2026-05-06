def update(self, message, content, branch=None, committer=None,
               author=None):
        """Update this file.

        :param str message: (required), commit message to describe the update
        :param str content: (required), content to update the file with
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
        if content and not isinstance(content, bytes):
            raise ValueError(  # (No coverage)
                'content must be a bytes object')  # (No coverage)

        json = None
        if message and content:
            content = b64encode(content).decode('utf-8')
            data = {'message': message, 'content': content, 'branch': branch,
                    'sha': self.sha,
                    'committer': validate_commmitter(committer),
                    'author': validate_commmitter(author)}
            self._remove_none(data)
            json = self._json(self._put(self._api, data=dumps(data)), 200)
            if 'content' in json and 'commit' in json:
                self.__init__(json['content'], self)
                json = Commit(json['commit'], self)
        return json