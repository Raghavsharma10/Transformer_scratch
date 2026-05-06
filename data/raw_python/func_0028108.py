def update_in_hdx(self, update_resources=True, update_resources_by_name=True,
                      remove_additional_resources=False, create_default_views=True, hxl_update=True):
        # type: (bool, bool, bool, bool, bool) -> None
        """Check if dataset exists in HDX and if so, update it

        Args:
            update_resources (bool): Whether to update resources. Defaults to True.
            update_resources_by_name (bool): Compare resource names rather than position in list. Defaults to True.
            remove_additional_resources (bool): Remove additional resources found in dataset. Defaults to False.
            create_default_views (bool): Whether to call package_create_default_resource_views. Defaults to True.
            hxl_update (bool): Whether to call package_hxl_update. Defaults to True.

        Returns:
            None
        """
        loaded = False
        if 'id' in self.data:
            self._check_existing_object('dataset', 'id')
            if self._dataset_load_from_hdx(self.data['id']):
                loaded = True
            else:
                logger.warning('Failed to load dataset with id %s' % self.data['id'])
        if not loaded:
            self._check_existing_object('dataset', 'name')
            if not self._dataset_load_from_hdx(self.data['name']):
                raise HDXError('No existing dataset to update!')
        self._dataset_merge_hdx_update(update_resources=update_resources,
                                       update_resources_by_name=update_resources_by_name,
                                       remove_additional_resources=remove_additional_resources,
                                       create_default_views=create_default_views,
                                       hxl_update=hxl_update)