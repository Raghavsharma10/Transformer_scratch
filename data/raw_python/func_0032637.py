def get(self, key, **kwargs):
        '''
        Fetch value at the given key
        kwargs can hold `recurse`, `wait` and `index` params
        '''
        return self._get('/'.join([self._endpoint, key]), payload=kwargs)