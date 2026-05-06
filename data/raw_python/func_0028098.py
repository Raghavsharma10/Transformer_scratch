def update_from_yaml(self, path=join('config', 'hdx_dataset_static.yml')):
        # type: (str) -> None
        """Update dataset metadata with static metadata from YAML file

        Args:
            path (str): Path to YAML dataset metadata. Defaults to config/hdx_dataset_static.yml.

        Returns:
            None
        """
        super(Dataset, self).update_from_yaml(path)
        self.separate_resources()