def read_from_hdx(identifier, configuration=None):
        # type: (str, Optional[Configuration]) -> Optional['Organization']
        """Reads the organization given by identifier from HDX and returns Organization object

        Args:
            identifier (str): Identifier of organization
            configuration (Optional[Configuration]): HDX configuration. Defaults to global configuration.

        Returns:
            Optional[Organization]: Organization object if successful read, None if not
        """

        organization = Organization(configuration=configuration)
        result = organization._load_from_hdx('organization', identifier)
        if result:
            return organization
        return None