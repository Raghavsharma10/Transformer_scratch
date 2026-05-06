def weekly_commit_count(self):
        """Returns the total commit counts.

        The dictionary returned has two entries: ``all`` and ``owner``. Each
        has a fifty-two element long list of commit counts. (Note: ``all``
        includes the owner.) ``d['all'][0]`` will be the oldest week,
        ``d['all'][51]`` will be the most recent.

        :returns: dict

        .. note:: All statistics methods may return a 202. If github3.py
            receives a 202 in this case, it will return an emtpy dictionary.
            You should give the API a moment to compose the data and then re
            -request it via this method.

        ..versionadded:: 0.7

        """
        url = self._build_url('stats', 'participation', base_url=self._api)
        resp = self._get(url)
        if resp.status_code == 202:
            return {}
        json = self._json(resp, 200)
        if json.get('ETag'):
            del json['ETag']
        if json.get('Last-Modified'):
            del json['Last-Modified']
        return json