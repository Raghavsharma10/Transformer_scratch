def update_from_json(self, path=join('config', 'hdx_dataset_static.json')):
        # type: (str) -> None
        """Update dataset metadata with static metadata from JSON file

        Args:
            path (str): Path to JSON dataset metadata. Defaults to config/hdx_dataset_static.json.

        Returns:
            None
        """
        super(Dataset, self).update_from_json(path)
        self.separate_resources()