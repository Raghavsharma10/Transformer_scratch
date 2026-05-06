def sample(self, hash, limit=None, offset=None):
        '''Return an object representing the sample identified by the input hash, or an empty object if that sample is not found'''

        uri = self._uris['sample'].format(hash)
        params = {'limit': limit, 'offset': offset}

        return self.get_parse(uri, params)