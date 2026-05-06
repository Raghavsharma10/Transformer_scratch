def get(self, identifier):
        """get provider by id"""
        for provider in self._providers:
            if provider.identifier == identifier:
                return provider
        return None