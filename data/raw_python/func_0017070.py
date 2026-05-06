def contents(self, path, ref=None):
        """Get the contents of the file pointed to by ``path``.

        If the path provided is actually a directory, you will receive a
        dictionary back of the form::

            {
                'filename.md': Contents(),  # Where Contents an instance
                'github.py': Contents(),
            }

        :param str path: (required), path to file, e.g.
            github3/repo.py
        :param str ref: (optional), the string name of a commit/branch/tag.
            Default: master
        :returns: :class:`Contents <github3.repos.contents.Contents>` or dict
            if successful, else None
        """
        url = self._build_url('contents', path, base_url=self._api)
        json = self._json(self._get(url, params={'ref': ref}), 200)
        if isinstance(json, dict):
            return Contents(json, self)
        elif isinstance(json, list):
            return dict((j.get('name'), Contents(j, self)) for j in json)
        return None