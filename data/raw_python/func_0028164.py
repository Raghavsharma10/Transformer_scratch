def get_all_for_resource(identifier, configuration=None):
        # type: (str, Optional[Configuration]) -> List['ResourceView']
        """Read all resource views for a resource given by identifier from HDX and returns list of ResourceView objects

        Args:
            identifier (str): Identifier of resource
            configuration (Optional[Configuration]): HDX configuration. Defaults to global configuration.

        Returns:
            List[ResourceView]: List of ResourceView objects
        """

        resourceview = ResourceView(configuration=configuration)
        success, result = resourceview._read_from_hdx('resource view', identifier, 'id', ResourceView.actions()['list'])
        resourceviews = list()
        if success:
            for resourceviewdict in result:
                resourceview = ResourceView(resourceviewdict, configuration=configuration)
                resourceviews.append(resourceview)
        return resourceviews