def get_HDX_code_from_location(location, locations=None, configuration=None):
        # type: (str, Optional[List[Dict]], Optional[Configuration]) -> Optional[str]
        """Get HDX code for location

        Args:
            location (str): Location for which to get HDX code
            locations (Optional[List[Dict]]): Valid locations list. Defaults to list downloaded from HDX.
            configuration (Optional[Configuration]): HDX configuration. Defaults to global configuration.

        Returns:
            Optional[str]: HDX code or None
        """
        if locations is None:
            locations = Locations.validlocations(configuration)
        locationupper = location.upper()
        for locdict in locations:
            locationcode = locdict['name'].upper()
            if locationupper == locationcode:
                return locationcode

        for locdict in locations:
            if locationupper == locdict['title'].upper():
                return locdict['name'].upper()
        return None