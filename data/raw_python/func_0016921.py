def merge(self, commit_message='', sha=None):
        """Merge this pull request.

        :param str commit_message: (optional), message to be used for the
            merge commit
        :returns: bool
        """
        parameters = {'commit_message': commit_message}
        if sha:
            parameters['sha'] = sha
        url = self._build_url('merge', base_url=self._api)
        json = self._json(self._put(url, data=dumps(parameters)), 200)
        self.merge_commit_sha = json['sha']
        return json['merged']