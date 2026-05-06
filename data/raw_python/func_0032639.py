def health(self, **kwargs):
        '''
        Support `node`, `service`, `check`, `state`
        '''
        if not len(kwargs):
            raise ValueError('no resource provided')
        for resource, name in kwargs.iteritems():
            endpoint = 'health/{}/{}'.format(resource, name)
        return self._get(endpoint)