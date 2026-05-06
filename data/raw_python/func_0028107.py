def _dataset_merge_hdx_update(self, update_resources, update_resources_by_name,
                                  remove_additional_resources, create_default_views, hxl_update):
        # type: (bool, bool, bool, bool, bool) -> None
        """Helper method to check if dataset or its resources exist and update them

        Args:
            update_resources (bool): Whether to update resources
            update_resources_by_name (bool): Compare resource names rather than position in list
            remove_additional_resources (bool): Remove additional resources found in dataset (if updating)
            create_default_views (bool): Whether to call package_create_default_resource_views.
            hxl_update (bool): Whether to call package_hxl_update.

        Returns:
            None
        """
        # 'old_data' here is the data we want to use for updating while 'data' is the data read from HDX
        merge_two_dictionaries(self.data, self.old_data)
        if 'resources' in self.data:
            del self.data['resources']
        updated_resources = self.old_data.get('resources', None)
        filestore_resources = list()
        if update_resources and updated_resources:
            ignore_fields = ['package_id']
            if update_resources_by_name:
                resource_names = set()
                for resource in self.resources:
                    resource_name = resource['name']
                    resource_names.add(resource_name)
                    for updated_resource in updated_resources:
                        if resource_name == updated_resource['name']:
                            logger.warning('Resource exists. Updating %s' % resource_name)
                            self._dataset_merge_filestore_resource(resource, updated_resource,
                                                                   filestore_resources, ignore_fields)
                            break
                updated_resource_names = set()
                for updated_resource in updated_resources:
                    updated_resource_name = updated_resource['name']
                    updated_resource_names.add(updated_resource_name)
                    if not updated_resource_name in resource_names:
                        self._dataset_merge_filestore_newresource(updated_resource, ignore_fields, filestore_resources)
                if remove_additional_resources:
                    resources_to_delete = list()
                    for i, resource in enumerate(self.resources):
                        resource_name = resource['name']
                        if resource_name not in updated_resource_names:
                            logger.warning('Removing additional resource %s!' % resource_name)
                            resources_to_delete.append(i)
                    for i in sorted(resources_to_delete, reverse=True):
                        del self.resources[i]

            else:  # update resources by position
                for i, updated_resource in enumerate(updated_resources):
                    if len(self.resources) > i:
                        updated_resource_name = updated_resource['name']
                        resource = self.resources[i]
                        resource_name = resource['name']
                        logger.warning('Resource exists. Updating %s' % resource_name)
                        if resource_name != updated_resource_name:
                            logger.warning('Changing resource name to: %s' % updated_resource_name)
                        self._dataset_merge_filestore_resource(resource, updated_resource,
                                                               filestore_resources, ignore_fields)
                    else:
                        self._dataset_merge_filestore_newresource(updated_resource, ignore_fields, filestore_resources)
                if remove_additional_resources:
                    resources_to_delete = list()
                    for i, resource in enumerate(self.resources):
                        if len(updated_resources) <= i:
                            logger.warning('Removing additional resource %s!' % resource['name'])
                            resources_to_delete.append(i)
                    for i in sorted(resources_to_delete, reverse=True):
                        del self.resources[i]

        if self.resources:
            self.data['resources'] = self._convert_hdxobjects(self.resources)
        ignore_field = self.configuration['dataset'].get('ignore_on_update')
        self.check_required_fields(ignore_fields=[ignore_field])
        self._save_to_hdx('update', 'id')
        self._add_filestore_resources(filestore_resources, create_default_views, hxl_update)