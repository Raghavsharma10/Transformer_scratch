def reorder_resource_views(self, resource_views):
        # type: (List[Union[ResourceView,Dict,str]]) -> None
        """Order resource views in resource.

        Args:
            resource_views (List[Union[ResourceView,Dict,str]]): A list of either resource view ids or resource views metadata from ResourceView objects or dictionaries

        Returns:
            None
        """
        if not isinstance(resource_views, list):
            raise HDXError('ResourceViews should be a list!')
        ids = list()
        for resource_view in resource_views:
            if isinstance(resource_view, str):
                resource_view_id = resource_view
            else:
                resource_view_id = resource_view['id']
            if is_valid_uuid(resource_view_id) is False:
                raise HDXError('%s is not a valid resource view id!' % resource_view)
            ids.append(resource_view_id)
        _, result = self._read_from_hdx('resource view', self.data['id'], 'id',
                                        ResourceView.actions()['reorder'], order=ids)