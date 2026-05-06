def symbol_for_ticker(self, ticker, field, provider, providers):
        '''Return an instance of *symboldata* containing
information about the data provider, the data provider ticker name
and the data provider field.'''
        provider = provider or settings.default_provider
        if provider:
            provider = providers.get(provider, None)
        return self.symboldata(ticker, field, provider)