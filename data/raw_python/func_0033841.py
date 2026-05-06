def find_observatories(self, match=None):
        """Query the LDR host for observatories. Use match to
        restrict returned observatories to those matching the
        regular expression.

        Example:

        >>> connection.find_observatories()
        ['AGHLT', 'G', 'GHLTV', 'GHLV', 'GHT', 'H', 'HL', 'HLT',
         'L', 'T', 'V', 'Z']
        >>> connection.find_observatories("H")
        ['H', 'HL', 'HLT']

        @type  match: L{str}
        @param match:
            name to match return observatories against

        @returns: L{list} of observatory prefixes
        """
        url = "%s/gwf.json" % _url_prefix
        response = self._requestresponse("GET", url)
        sitelist = sorted(set(decode(response.read())))
        if match:
            regmatch = re.compile(match)
            sitelist = [site for site in sitelist if regmatch.search(site)]
        return sitelist