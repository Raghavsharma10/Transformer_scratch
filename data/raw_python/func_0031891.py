def create_stash(self, payload, path=None):
        """
        Create a stash. (JSON document)
        """
        if path:
            self._request('POST', '/stashes/{}'.format(path),
                          json=payload)
        else:
            self._request('POST', '/stashes', json=payload)
        return True