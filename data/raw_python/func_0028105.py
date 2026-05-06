def _dataset_merge_filestore_newresource(self, new_resource, ignore_fields, filestore_resources):
        # type: (hdx.data.Resource, List[str], List[hdx.data.Resource]) -> None
        """Helper method to add new resource from dataset including filestore.

        Args:
            new_resource (hdx.data.Resource): New resource from dataset
            ignore_fields (List[str]): List of fields to ignore when checking resource
            filestore_resources (List[hdx.data.Resource]): List of resources that use filestore (to be appended to)

        Returns:
            None
        """
        new_resource.check_required_fields(ignore_fields=ignore_fields)
        self.resources.append(new_resource)
        if new_resource.get_file_to_upload():
            filestore_resources.append(new_resource)
            new_resource['url'] = Dataset.temporary_url