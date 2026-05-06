def create_blob(self, content, encoding):
        """Create a blob with ``content``.

        :param str content: (required), content of the blob
        :param str encoding: (required), ('base64', 'utf-8')
        :returns: string of the SHA returned
        """
        sha = ''
        if encoding in ('base64', 'utf-8'):
            url = self._build_url('git', 'blobs', base_url=self._api)
            data = {'content': content, 'encoding': encoding}
            json = self._json(self._post(url, data=data), 201)
            if json:
                sha = json.get('sha')
        return sha