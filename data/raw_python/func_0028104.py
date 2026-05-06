def _dataset_merge_filestore_resource(self, resource, updated_resource, filestore_resources, ignore_fields):
        # type: (hdx.data.Resource, hdx.data.Resource, List[hdx.data.Resource], List[str]) -> None
        """Helper method to merge updated resource from dataset into HDX resource read from HDX including filestore.

        Args:
            resource (hdx.data.Resource): Resource read from HDX
            updated_resource (hdx.data.Resource): Updated resource from dataset
            filestore_resources (List[hdx.data.Resource]): List of resources that use filestore (to be appended to)
            ignore_fields (List[str]): List of fields to ignore when checking resource

        Returns:
            None
        """
        if updated_resource.get_file_to_upload():
            resource.set_file_to_upload(updated_resource.get_file_to_upload())
            filestore_resources.append(resource)
        merge_two_dictionaries(resource, updated_resource)
        resource.check_required_fields(ignore_fields=ignore_fields)
        if resource.get_file_to_upload():
            resource['url'] = Dataset.temporary_url