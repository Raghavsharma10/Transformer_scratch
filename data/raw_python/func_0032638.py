def set(self, key, value, **kwargs):
        '''
        Store a new value at the given key
        kwargs can hold `cas` and `flags` params
        '''
        return requests.put(
            '{}/{}/kv/{}'.format(
                self.master, pyconsul.__consul_api_version__, key),
            data=value,
            params=kwargs
        )