def ratelimit_remaining(self):
        """Number of requests before GitHub imposes a ratelimit.

        :returns: int
        """
        json = self._json(self._get(self._github_url + '/rate_limit'), 200)
        core = json.get('resources', {}).get('core', {})
        self._remaining = core.get('remaining', 0)
        return self._remaining