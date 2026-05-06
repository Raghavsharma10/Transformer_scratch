def samples(self, anystring, limit=None, offset=None, sortby=None):
        '''Return an object representing the samples identified by the input domain, IP, or URL'''

        uri = self._uris['samples'].format(anystring)
        params = {'limit': limit, 'offset': offset, 'sortby': sortby}

        return self.get_parse(uri, params)