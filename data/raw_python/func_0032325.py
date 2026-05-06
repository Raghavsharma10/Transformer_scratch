def financials(self, security):
        """
        get financials:
        google finance provide annual and quanter financials, if annual is true, we will use annual data
        Up to four lastest year/quanter data will be provided by google
        Refer to page as an example: http://www.google.com/finance?q=TSE:CVG&fstype=ii
        """
        try:
            url = 'http://www.google.com/finance?q=%s&fstype=ii' % security
            try:
                page = self._request(url).read()
            except UfException as ufExcep:
                # if symol is not right, will get 400
                if Errors.NETWORK_400_ERROR == ufExcep.getCode:
                    raise UfException(Errors.STOCK_SYMBOL_ERROR, "Can find data for stock %s, security error?" % security)
                raise ufExcep

            bPage = BeautifulSoup(page)
            target = bPage.find(id='incinterimdiv')

            keyTimeValue = {}
            # ugly do...while
            i = 0
            while True:
                self._parseTarget(target, keyTimeValue)

                if i < 5:
                    i += 1
                    target = target.nextSibling
                    # ugly beautiful soap...
                    if '\n' == target:
                        target = target.nextSibling
                else:
                    break

            return keyTimeValue

        except BaseException:
            raise UfException(Errors.UNKNOWN_ERROR, "Unknown Error in GoogleFinance.getHistoricalPrices %s" % traceback.format_exc())