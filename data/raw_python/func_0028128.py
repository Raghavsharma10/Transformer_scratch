def add_country_locations(self, countries, locations=None, use_live=True):
        # type: (List[str], Optional[List[str]], bool) -> bool
        """Add a list of countries. If iso 3 codes are not provided, values are parsed and where they are valid country
        names, converted to iso 3 codes. If any country is already added, it is ignored.

        Args:
            countries (List[str]): list of countries to add
            locations (Optional[List[str]]): Valid locations list. Defaults to list downloaded from HDX.
            use_live (bool): Try to get use latest country data from web rather than file in package. Defaults to True.

        Returns:
            bool: True if all countries added or False if any already present.
        """
        allcountriesadded = True
        for country in countries:
            if not self.add_country_location(country, locations=locations, use_live=use_live):
                allcountriesadded = False
        return allcountriesadded