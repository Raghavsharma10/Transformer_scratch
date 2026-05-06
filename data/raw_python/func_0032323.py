def get_all(self, security):
        """
        Get all available quote data for the given ticker security.
        Returns a dictionary.
        """
        url = 'http://www.google.com/finance?q=%s' % security
        page = self._request(url)

        soup = BeautifulSoup(page)
        snapData = soup.find("table", {"class": "snap-data"})
        if snapData is None:
            raise UfException(Errors.STOCK_SYMBOL_ERROR, "Can find data for stock %s, security error?" % security)
        data = {}
        for row in snapData.findAll('tr'):
            keyTd, valTd = row.findAll('td')
            data[keyTd.getText()] = valTd.getText()

        return data