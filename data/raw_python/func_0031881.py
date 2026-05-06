def get_check(self, check):
        """
        Returns a check.
        """
        data = self._request('GET', '/checks/{}'.format(check))
        return data.json()