def load(self, providers, symbols, start, end, logger, backend, **kwargs):
        '''Load symbols data.

        :keyword providers: Dictionary of registered data providers.
        :keyword symbols: list of symbols to load.
        :keyword start: start date.
        :keyword end: end date.
        :keyword logger: instance of :class:`logging.Logger` or ``None``.
        :keyword backend: :class:`dynts.TimeSeries` backend name.

        There is no need to override this function, just use one
        the three hooks available.
        '''
        # Preconditioning on dates
        logger = logger or logging.getLogger(self.__class__.__name__)
        start, end = self.dates(start, end)
        data = {}
        for sym in symbols:
            # Get ticker, field and provider
            symbol = self.parse_symbol(sym, providers)
            provider = symbol.provider
            if not provider:
                raise MissingDataProvider(
                    'data provider for %s not available' % sym
                )
            pre = self.preprocess(symbol, start, end, logger, backend, **kwargs)
            if pre.intervals:
                result = None
                for st, en in pre.intervals:
                    logger.info('Loading %s from %s. From %s to %s',
                                symbol.ticker, provider, st, en)
                    res = provider.load(symbol, st, en, logger, backend,
                                        **kwargs)
                    if result is None:
                        result = res
                    else:
                        result.update(res)
            else:
                result = pre.result
            # onresult hook
            result = self.onresult(symbol, result, logger, backend, **kwargs)
            data[sym] = result
        # last hook
        return self.onfinishload(data, logger, backend, **kwargs)