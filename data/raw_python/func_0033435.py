def to_url(self, site='amazon', country='us'):
        """Generate a link to an online book site.

        Args:
            site (str): Site to create link to
            country (str): Country specific version of ``site``

        Returns:
            ``str``: URL on ``site`` for book

        Raises:
            SiteError: Unknown site value
            CountryError: Unknown country value

        """
        try:
            try:
                url, tlds = URL_MAP[site]
            except ValueError:
                tlds = None
                url = URL_MAP[site]
        except KeyError:
            raise SiteError(site)
        inject = {'isbn': self._isbn}
        if tlds:
            if country not in tlds:
                raise CountryError(country)
            tld = tlds[country]
            if not tld:
                tld = country
            inject['tld'] = tld
        return url % inject