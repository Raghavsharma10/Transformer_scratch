def tag(self, tag):
        """Get a release by tag
        """
        url = '%s/tags/%s' % (self, tag)
        response = self.http.get(url, auth=self.auth)
        response.raise_for_status()
        return response.json()