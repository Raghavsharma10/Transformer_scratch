def add_update_resource_views(self, resource_views):
        # type: (List[Union[ResourceView,Dict]]) -> None
        """Add new or update existing resource views in resource with new metadata.

        Args:
            resource_views (List[Union[ResourceView,Dict]]): A list of resource views metadata from ResourceView objects or dictionaries

        Returns:
            None
        """
        if not isinstance(resource_views, list):
            raise HDXError('ResourceViews should be a list!')
        for resource_view in resource_views:
            self.add_update_resource_view(resource_view)