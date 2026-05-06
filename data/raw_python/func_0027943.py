def validlocations(configuration=None):
        # type: () -> List[Dict]
        """
        Read valid locations from HDX

        Args:
            configuration (Optional[Configuration]): HDX configuration. Defaults to global configuration.

        Returns:
            List[Dict]: A list of valid locations
        """
        if Locations._validlocations is None:
            if configuration is None:
                configuration = Configuration.read()
            Locations._validlocations = configuration.call_remoteckan('group_list', {'all_fields': True})
        return Locations._validlocations