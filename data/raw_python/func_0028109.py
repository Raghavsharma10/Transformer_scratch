def create_in_hdx(self, allow_no_resources=False, update_resources=True, update_resources_by_name=True,
                      remove_additional_resources=False, create_default_views=True, hxl_update=True):
        # type: (bool, bool, bool, bool, bool, bool) -> None
        """Check if dataset exists in HDX and if so, update it, otherwise create it

        Args:
            allow_no_resources (bool): Whether to allow no resources. Defaults to False.
            update_resources (bool): Whether to update resources (if updating). Defaults to True.
            update_resources_by_name (bool): Compare resource names rather than position in list. Defaults to True.
            remove_additional_resources (bool): Remove additional resources found in dataset (if updating). Defaults to False.
            create_default_views (bool): Whether to call package_create_default_resource_views (if updating). Defaults to True.
            hxl_update (bool): Whether to call package_hxl_update. Defaults to True.

        Returns:
            None
        """
        self.check_required_fields(allow_no_resources=allow_no_resources)
        loadedid = None
        if 'id' in self.data:
            if self._dataset_load_from_hdx(self.data['id']):
                loadedid = self.data['id']
            else:
                logger.warning('Failed to load dataset with id %s' % self.data['id'])
        if not loadedid:
            if self._dataset_load_from_hdx(self.data['name']):
                loadedid = self.data['name']
        if loadedid:
            logger.warning('Dataset exists. Updating %s' % loadedid)
            self._dataset_merge_hdx_update(update_resources=update_resources,
                                           update_resources_by_name=update_resources_by_name,
                                           remove_additional_resources=remove_additional_resources,
                                           create_default_views=create_default_views,
                                           hxl_update=hxl_update)
            return

        filestore_resources = list()
        if self.resources:
            ignore_fields = ['package_id']
            for resource in self.resources:
                resource.check_required_fields(ignore_fields=ignore_fields)
                if resource.get_file_to_upload():
                    filestore_resources.append(resource)
                    resource['url'] = Dataset.temporary_url
            self.data['resources'] = self._convert_hdxobjects(self.resources)
        self._save_to_hdx('create', 'name')
        self._add_filestore_resources(filestore_resources, False, hxl_update)