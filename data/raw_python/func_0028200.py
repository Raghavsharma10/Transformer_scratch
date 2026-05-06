def add_dataset(self, dataset, datasets_to_check=None):
        # type: (Union[hdx.data.dataset.Dataset,Dict,str], List[hdx.data.dataset.Dataset]) -> bool
        """Add a dataset

        Args:
            dataset (Union[Dataset,Dict,str]): Either a dataset id or dataset metadata either from a Dataset object or a dictionary
            datasets_to_check (List[Dataset]): List of datasets against which to check existence of dataset. Defaults to datasets in showcase.

        Returns:
            bool: True if the dataset was added, False if already present
        """
        showcase_dataset = self._get_showcase_dataset_dict(dataset)
        if datasets_to_check is None:
            datasets_to_check = self.get_datasets()
        for dataset in datasets_to_check:
            if showcase_dataset['package_id'] == dataset['id']:
                return False
        self._write_to_hdx('associate', showcase_dataset, 'package_id')
        return True