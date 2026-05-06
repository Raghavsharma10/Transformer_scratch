def read_from_hdx(identifier, configuration=None):
        # type: (str, Optional[Configuration]) -> Optional['Resource']
        """Reads the resource given by identifier from HDX and returns Resource object

        Args:
            identifier (str): Identifier of resource
            configuration (Optional[Configuration]): HDX configuration. Defaults to global configuration.

        Returns:
            Optional[Resource]: Resource object if successful read, None if not
        """

        if is_valid_uuid(identifier) is False:
            raise HDXError('%s is not a valid resource id!' % identifier)
        resource = Resource(configuration=configuration)
        result = resource._load_from_hdx('resource', identifier)
        if result:
            return resource
        return None