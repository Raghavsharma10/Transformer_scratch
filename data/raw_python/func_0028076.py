def delete_resource_view(self, resource_view):
        # type: (Union[ResourceView,Dict,str]) -> None
        """Delete a resource view from the resource and HDX

        Args:
            resource_view (Union[ResourceView,Dict,str]): Either a resource view id or resource view metadata either from a ResourceView object or a dictionary

        Returns:
            None
        """
        if isinstance(resource_view, str):
            if is_valid_uuid(resource_view) is False:
                raise HDXError('%s is not a valid resource view id!' % resource_view)
            resource_view = ResourceView({'id': resource_view}, configuration=self.configuration)
        else:
            resource_view = self._get_resource_view(resource_view)
            if 'id' not in resource_view:
                found = False
                title = resource_view.get('title')
                for rv in self.get_resource_views():
                    if resource_view['title'] == rv['title']:
                        resource_view = rv
                        found = True
                        break
                if not found:
                    raise HDXError('No resource views have title %s in this resource!' % title)
        resource_view.delete_from_hdx()