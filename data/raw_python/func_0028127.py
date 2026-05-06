def add_country_location(self, country, exact=True, locations=None, use_live=True):
        # type: (str, bool, Optional[List[str]], bool) -> bool
        """Add a country. If an iso 3 code is not provided, value is parsed and if it is a valid country name,
        converted to an iso 3 code. If the country is already added, it is ignored.

        Args:
            country (str): Country to add
            exact (bool): True for exact matching or False to allow fuzzy matching. Defaults to True.
            locations (Optional[List[str]]): Valid locations list. Defaults to list downloaded from HDX.
            use_live (bool): Try to get use latest country data from web rather than file in package. Defaults to True.

        Returns:
            bool: True if country added or False if country already present
        """
        iso3, match = Country.get_iso3_country_code_fuzzy(country, use_live=use_live)
        if iso3 is None:
            raise HDXError('Country: %s - cannot find iso3 code!' % country)
        return self.add_other_location(iso3, exact=exact,
                                       alterror='Country: %s with iso3: %s could not be found in HDX list!' %
                                                (country, iso3),
                                       locations=locations)