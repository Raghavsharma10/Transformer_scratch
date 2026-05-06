def add_update_resource(self, resource, ignore_datasetid=False):
        # type: (Union[hdx.data.resource.Resource,Dict,str], bool) -> None
        """Add new or update existing resource in dataset with new metadata

        Args:
            resource (Union[hdx.data.resource.Resource,Dict,str]): Either resource id or resource metadata from a Resource object or a dictionary
            ignore_datasetid (bool): Whether to ignore dataset id in the resource

        Returns:
            None
        """
        resource = self._get_resource_from_obj(resource)
        if 'package_id' in resource:
            if not ignore_datasetid:
                raise HDXError('Resource %s being added already has a dataset id!' % (resource['name']))
        resource.check_url_filetoupload()
        resource_updated = self._addupdate_hdxobject(self.resources, 'name', resource)
        if resource.get_file_to_upload():
            resource_updated.set_file_to_upload(resource.get_file_to_upload())