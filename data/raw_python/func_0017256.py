def unregister(self, provider):
        '''Unregister an existing data provider.

        *provider* must be an instance of DataProvider.
        If provider name is already available, it will be replaced.
        '''
        if isinstance(provider, type):
            provider = provider()
        if isinstance(provider, DataProvider):
            provider = provider.code
        return self.pop(str(provider).upper(), None)