def add_datasets(self, datasets, datasets_to_check=None):
        # type: (List[Union[hdx.data.dataset.Dataset,Dict,str]], List[hdx.data.dataset.Dataset]) -> bool
        """Add multiple datasets

        Args:
            datasets (List[Union[Dataset,Dict,str]]): A list of either dataset ids or dataset metadata from Dataset objects or dictionaries
            datasets_to_check (List[Dataset]): List of datasets against which to check existence of dataset. Defaults to datasets in showcase.

        Returns:
            bool: True if all datasets added or False if any already present
        """
        if datasets_to_check is None:
            datasets_to_check = self.get_datasets()
        alldatasetsadded = True
        for dataset in datasets:
            if not self.add_dataset(dataset, datasets_to_check=datasets_to_check):
                alldatasetsadded = False
        return alldatasetsadded