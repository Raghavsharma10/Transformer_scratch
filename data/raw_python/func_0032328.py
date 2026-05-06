def ticks(self, security, start, end):
        """
        Get tick prices for the given ticker security.
        @security: stock security
        @interval: interval in mins(google finance only support query till 1 min)
        @start: start date(YYYYMMDD)
        @end: end date(YYYYMMDD)

        start and end is disabled since only 15 days data will show

        @Returns a nested list.
        """
        period = 1
        # url = 'http://www.google.com/finance/getprices?q=%s&i=%s&p=%sd&f=d,o,h,l,c,v&ts=%s' % (security, interval, period, start)
        url = 'http://www.google.com/finance/getprices?q=%s&i=61&p=%sd&f=d,o,h,l,c,v' % (security.symbol, period)
        LOG.debug('fetching {0}'.format(url))
        try:
            response = self._request(url)
        except UfException as ufExcep:
            # if symol is not right, will get 400
            if Errors.NETWORK_400_ERROR == ufExcep.getCode:
                raise UfException(Errors.STOCK_SYMBOL_ERROR, "Can find data for stock %s, security error?" % security)
            raise ufExcep

        # use csv reader here
        days = response.text.split('\n')[7:]  # first 7 line is document
        # sample values:'a1316784600,31.41,31.5,31.4,31.43,150911'
        values = [day.split(',') for day in days if len(day.split(',')) >= 6]
        for value in values:
            yield json.dumps({'date': value[0][1:].strip(),
                'close': value[1].strip(),
                'high': value[2].strip(),
                'low': value[3].strip(),
                'open': value[4].strip(),
                'volume': value[5].strip()})