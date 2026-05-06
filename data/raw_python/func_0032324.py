def quotes(self, security, start, end):
        """
        Get historical prices for the given ticker security.
        Date format is 'YYYYMMDD'

        Returns a nested list.
        """
        try:
            url = 'http://www.google.com/finance/historical?q=%s&startdate=%s&enddate=%s&output=csv' % (security.symbol, start, end)
            try:
                page = self._request(url)
            except UfException as ufExcep:
                # if symol is not right, will get 400
                if Errors.NETWORK_400_ERROR == ufExcep.getCode:
                    raise UfException(Errors.STOCK_SYMBOL_ERROR, "Can find data for stock %s, security error?" % security)
                raise ufExcep

            days = page.readlines()
            values = [day.split(',') for day in days]
            # sample values:[['Date', 'Open', 'High', 'Low', 'Close', 'Volume'], \
            #              ['2009-12-31', '112.77', '112.80', '111.39', '111.44', '90637900']...]
            for value in values[1:]:
                date = convertGoogCSVDate(value[0])
                try:
                    yield Quote(date,
                                      value[1].strip(),
                                      value[2].strip(),
                                      value[3].strip(),
                                      value[4].strip(),
                                      value[5].strip(),
                                      None)
                except Exception:
                    LOG.warning("Exception when processing %s at date %s for value %s" % (security, date, value))

        except BaseException:
            raise UfException(Errors.UNKNOWN_ERROR, "Unknown Error in GoogleFinance.getHistoricalPrices %s" % traceback.format_exc())