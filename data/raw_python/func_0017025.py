def octocat(self, say=None):
        """Returns an easter egg of the API.

        :params str say: (optional), pass in what you'd like Octocat to say
        :returns: ascii art of Octocat
        """
        url = self._build_url('octocat')
        req = self._get(url, params={'s': say})
        return req.content if req.ok else ''