def _get_resource_view(self, resource_view):
        # type: (Union[ResourceView,Dict]) -> ResourceView
        """Get resource view id

        Args:
            resource_view (Union[ResourceView,Dict]): ResourceView metadata from a ResourceView object or dictionary

        Returns:
            ResourceView: ResourceView object
        """
        if isinstance(resource_view, dict):
            resource_view = ResourceView(resource_view, configuration=self.configuration)
        if isinstance(resource_view, ResourceView):
            return resource_view
        raise HDXError('Type %s is not a valid resource view!' % type(resource_view).__name__)