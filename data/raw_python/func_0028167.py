def copy(self, resource_view):
        # type: (Union[ResourceView,Dict,str]) -> None
        """Copies all fields except id, resource_id and package_id from another resource view.

        Args:
            resource_view (Union[ResourceView,Dict,str]): Either a resource view id or resource view metadata either from a ResourceView object or a dictionary

        Returns:
            None
        """
        if isinstance(resource_view, str):
            if is_valid_uuid(resource_view) is False:
                raise HDXError('%s is not a valid resource view id!' % resource_view)
            resource_view = ResourceView.read_from_hdx(resource_view)
        if not isinstance(resource_view, dict) and not isinstance(resource_view, ResourceView):
            raise HDXError('%s is not a valid resource view!' % resource_view)
        for key in resource_view:
            if key not in ('id', 'resource_id', 'package_id'):
                self.data[key] = resource_view[key]