def parse_symbol(self, symbol, providers):
        '''Parse a symbol to obtain information regarding ticker,
        field and provider. Must return an instance of :attr:`symboldata`.

        :keyword symbol: string associated with market data to load.
        :keyword providers: dictionary of :class:`dynts.data.DataProvider`
                            instances available.

        For example::

            intc
            intc:open
            intc:volume:google
            intc:google

        are all valid inputs returning a :class:`SymbolData` instance with
        the following triplet of information::

            intc,None,yahoo
            intc,open,yahoo
            intc,volume,google
            intc,None,google

        assuming ``yahoo`` is the provider in
        :attr:`dynts.conf.Settings.default_provider`.

        This function is called before retrieving data.
        '''
        separator = settings.field_separator
        symbol = str(symbol)
        bits = symbol.split(separator)
        pnames = providers.keys()
        ticker = symbol
        provider = None
        field = None
        if len(bits) == 2:
            ticker = bits[0]
            if bits[1] in pnames:
                provider = bits[1]
            else:
                field = bits[1]
        elif len(bits) == 3:
            ticker = bits[0]
            if bits[1] in pnames:
                provider = bits[1]
                field = bits[2]
            elif bits[2] in pnames:
                provider = bits[2]
                field = bits[1]
            else:
                raise BadSymbol(
                        'Could not parse %s. Unrecognized provider.' % symbol)
        elif len(bits) > 3:
            raise BadSymbol('Could not parse %s.' % symbol)

        return self.symbol_for_ticker(ticker, field, provider, providers)