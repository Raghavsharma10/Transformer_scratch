def read_from_hdx(identifier, configuration=None):
        # type: (str, Optional[Configuration]) -> Optional['Showcase']
        """Reads the showcase given by identifier from HDX and returns Showcase object

        Args:
            identifier (str): Identifier of showcase
            configuration (Optional[Configuration]): HDX configuration. Defaults to global configuration.

        Returns:
            Optional[Showcase]: Showcase object if successful read, None if not
        """

        showcase = Showcase(configuration=configuration)
        result = showcase._load_from_hdx('showcase', identifier)
        if result:
            return showcase
        return None