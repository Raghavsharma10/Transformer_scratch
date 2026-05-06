def read_from_hdx(identifier, configuration=None):
        # type: (str, Optional[Configuration]) -> Optional['ResourceView']
        """Reads the resource view given by identifier from HDX and returns ResourceView object

        Args:
            identifier (str): Identifier of resource view
            configuration (Optional[Configuration]): HDX configuration. Defaults to global configuration.

        Returns:
            Optional[ResourceView]: ResourceView object if successful read, None if not
        """

        resourceview = ResourceView(configuration=configuration)
        result = resourceview._load_from_hdx('resource view', identifier)
        if result:
            return resourceview
        return None