def add_region_location(self, region, locations=None, use_live=True):
        # type: (str, Optional[List[str]], bool) -> bool
        """Add all countries in a region. If a 3 digit UNStats M49 region code is not provided, value is parsed as a
        region name. If any country is already added, it is ignored.

        Args:
            region (str): M49 region, intermediate region or subregion to add
            locations (Optional[List[str]]): Valid locations list. Defaults to list downloaded from HDX.
            use_live (bool): Try to get use latest country data from web rather than file in package. Defaults to True.

        Returns:
            bool: True if all countries in region added or False if any already present.
        """
        return self.add_country_locations(Country.get_countries_in_region(region, exception=HDXError,
                                                                          use_live=use_live), locations=locations)