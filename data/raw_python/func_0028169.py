def read_from_hdx(identifier, configuration=None):
        # type: (str, Optional[Configuration]) -> Optional['User']
        """Reads the user given by identifier from HDX and returns User object

        Args:
            identifier (str): Identifier of user
            configuration (Optional[Configuration]): HDX configuration. Defaults to global configuration.

        Returns:
            Optional[User]: User object if successful read, None if not
        """

        user = User(configuration=configuration)
        result = user._load_from_hdx('user', identifier)
        if result:
            return user
        return None