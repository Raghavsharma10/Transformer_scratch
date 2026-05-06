def register(self, provider):
        '''Register a new data provider. *provider* must be an instance of
    DataProvider. If provider name is already available, it will be replaced.'''
        if isinstance(provider,type):
            provider = provider()
        self[provider.code] = provider