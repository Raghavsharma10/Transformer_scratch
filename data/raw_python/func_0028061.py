def get_dataset(self):
        # type: () -> hdx.data.dataset.Dataset
        """Return dataset containing this resource

        Returns:
            hdx.data.dataset.Dataset: Dataset containing this resource
        """
        package_id = self.data.get('package_id')
        if package_id is None:
            raise HDXError('Resource has no package id!')
        return hdx.data.dataset.Dataset.read_from_hdx(package_id)