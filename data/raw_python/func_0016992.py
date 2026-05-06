def create_gist(self, description, files, public=True):
        """Create a new gist.

        If no login was provided, it will be anonymous.

        :param str description: (required), description of gist
        :param dict files: (required), file names with associated dictionaries
            for content, e.g. ``{'spam.txt': {'content': 'File contents
            ...'}}``
        :param bool public: (optional), make the gist public if True
        :returns: :class:`Gist <github3.gists.Gist>`
        """
        new_gist = {'description': description, 'public': public,
                    'files': files}
        url = self._build_url('gists')
        json = self._json(self._post(url, data=new_gist), 201)
        return Gist(json, self) if json else None