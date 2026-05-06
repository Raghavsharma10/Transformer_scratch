def edit(self, description='', files={}):
        """Edit this gist.

        :param str description: (optional), description of the gist
        :param dict files: (optional), files that make up this gist; the
            key(s) should be the file name(s) and the values should be another
            (optional) dictionary with (optional) keys: 'content' and
            'filename' where the former is the content of the file and the
            latter is the new name of the file.
        :returns: bool -- whether the edit was successful

        """
        data = {}
        json = None
        if description:
            data['description'] = description
        if files:
            data['files'] = files
        if data:
            json = self._json(self._patch(self._api, data=dumps(data)), 200)
        if json:
            self._update_(json)
            return True
        return False