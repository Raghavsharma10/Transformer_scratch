def _add_filestore_resources(self, filestore_resources, create_default_views, hxl_update):
        # type: (List[hdx.data.Resource], bool, bool) -> None
        """Helper method to create files in filestore by updating resources.

        Args:
            filestore_resources (List[hdx.data.Resource]): List of resources that use filestore (to be appended to)
            create_default_views (bool): Whether to call package_create_default_resource_views.
            hxl_update (bool): Whether to call package_hxl_update.

        Returns:
            None
        """
        for resource in filestore_resources:
            for created_resource in self.data['resources']:
                if resource['name'] == created_resource['name']:
                    merge_two_dictionaries(resource.data, created_resource)
                    del resource['url']
                    resource.update_in_hdx()
                    merge_two_dictionaries(created_resource, resource.data)
                    break
        self.init_resources()
        self.separate_resources()
        if create_default_views:
            self.create_default_views()
        if hxl_update:
            self.hxl_update()