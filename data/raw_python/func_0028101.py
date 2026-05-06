def _dataset_create_resources(self):
        # type: () -> None
        """Creates resource objects in dataset
        """

        if 'resources' in self.data:
            self.old_data['resources'] = self._copy_hdxobjects(self.resources, hdx.data.resource.Resource, 'file_to_upload')
            self.init_resources()
            self.separate_resources()