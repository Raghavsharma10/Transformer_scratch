def set_quickchart_resource(self, resource):
        # type: (Union[hdx.data.resource.Resource,Dict,str,int]) -> bool
        """Set the resource that will be used for displaying QuickCharts in dataset preview

        Args:
            resource (Union[hdx.data.resource.Resource,Dict,str,int]): Either resource id or name, resource metadata from a Resource object or a dictionary or position

        Returns:
            bool: Returns True if resource for QuickCharts in dataset preview set or False if not
        """
        if isinstance(resource, int) and not isinstance(resource, bool):
            resource = self.get_resources()[resource]
        if isinstance(resource, hdx.data.resource.Resource) or isinstance(resource, dict):
            res = resource.get('id')
            if res is None:
                resource = resource['name']
            else:
                resource = res
        elif not isinstance(resource, str):
            raise hdx.data.hdxobject.HDXError('Resource id cannot be found in type %s!' % type(resource).__name__)
        if is_valid_uuid(resource) is True:
            search = 'id'
        else:
            search = 'name'
        changed = False
        for dataset_resource in self.resources:
            if dataset_resource[search] == resource:
                dataset_resource.enable_dataset_preview()
                self.preview_resource()
                changed = True
            else:
                dataset_resource.disable_dataset_preview()
        return changed