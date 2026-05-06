def get_HDX_code_from_location_partial(location, locations=None, configuration=None):
        # type: (str, Optional[List[Dict]], Optional[Configuration]) -> Tuple[Optional[str], bool]
        """Get HDX code for location

        Args:
            location (str): Location for which to get HDX code
            locations (Optional[List[Dict]]): Valid locations list. Defaults to list downloaded from HDX.
            configuration (Optional[Configuration]): HDX configuration. Defaults to global configuration.

        Returns:
            Tuple[Optional[str], bool]: HDX code and if the match is exact or (None, False) for no match
        """
        hdx_code = Locations.get_HDX_code_from_location(location, locations, configuration)

        if hdx_code is not None:
            return hdx_code, True

        if locations is None:
            locations = Locations.validlocations(configuration)
        locationupper = location.upper()
        for locdict in locations:
            locationname = locdict['title'].upper()
            if locationupper in locationname or locationname in locationupper:
                return locdict['name'].upper(), False

        return None, False