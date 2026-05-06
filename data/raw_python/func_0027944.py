def get_location_from_HDX_code(code, locations=None, configuration=None):
        # type: (str, Optional[List[Dict]], Optional[Configuration]) -> Optional[str]
        """Get location from HDX location code

        Args:
            code (str): code for which to get location name
            locations (Optional[List[Dict]]): Valid locations list. Defaults to list downloaded from HDX.
            configuration (Optional[Configuration]): HDX configuration. Defaults to global configuration.

        Returns:
            Optional[str]: location name
        """
        if locations is None:
            locations = Locations.validlocations(configuration)
        for locdict in locations:
            if code.upper() == locdict['name'].upper():
                return locdict['title']
        return None