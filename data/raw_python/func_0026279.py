def query(self, address, acceptlanguage=None, limit=20,
              countrycodes=None):
        """
        Issue a geocoding query for *address* to the
        Nominatim instance and return the decoded results

        :param address: a query string with an address
                        or presumed parts of an address
        :type address: str or (if python2) unicode
        :param acceptlanguage: rfc2616 language code
        :type acceptlanguage: str or None
        :param limit: limit the number of results
        :type limit: int or None
        :param countrycodes: restrict the search to countries
             given by their ISO 3166-1alpha2 codes (cf.
             https://en.wikipedia.org/wiki/ISO_3166-1_alpha-2 )
        :type countrycodes: str iterable
        :returns: a list of search results (each a dict)
        :rtype: list or None
        """
        url = self.url + '&q=' + quote_plus(address)
        if acceptlanguage:
            url += '&accept-language=' + acceptlanguage
        if limit:
            url += '&limit=' + str(limit)
        if countrycodes:
            url += '&countrycodes=' + ','.join(countrycodes)
        return self.request(url)