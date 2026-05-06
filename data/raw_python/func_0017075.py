def create_file(self, path, message, content, branch=None,
                    committer=None, author=None):
        """Create a file in this repository.

        See also: http://developer.github.com/v3/repos/contents/#create-a-file

        :param str path: (required), path of the file in the repository
        :param str message: (required), commit message
        :param bytes content: (required), the actual data in the file
        :param str branch: (optional), branch to create the commit on.
            Defaults to the default branch of the repository
        :param dict committer: (optional), if no information is given the
            authenticated user's information will be used. You must specify
            both a name and email.
        :param dict author: (optional), if omitted this will be filled in with
            committer information. If passed, you must specify both a name and
            email.
        :returns: {
            'content': :class:`Contents <github3.repos.contents.Contents>`:,
            'commit': :class:`Commit <github3.git.Commit>`}

        """
        if content and not isinstance(content, bytes):
            raise ValueError(  # (No coverage)
                'content must be a bytes object')  # (No coverage)

        json = None
        if path and message and content:
            url = self._build_url('contents', path, base_url=self._api)
            content = b64encode(content).decode('utf-8')
            data = {'message': message, 'content': content, 'branch': branch,
                    'committer': validate_commmitter(committer),
                    'author': validate_commmitter(author)}
            self._remove_none(data)
            json = self._json(self._put(url, data=dumps(data)), 201)
            if 'content' in json and 'commit' in json:
                json['content'] = Contents(json['content'], self)
                json['commit'] = Commit(json['commit'], self)
        return json