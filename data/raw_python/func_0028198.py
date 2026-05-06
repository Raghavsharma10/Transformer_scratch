def get_datasets(self):
        # type: () -> List[hdx.data.dataset.Dataset]
        """Get any datasets in the showcase

        Returns:
            List[Dataset]: List of datasets
        """
        assoc_result, datasets_dicts = self._read_from_hdx('showcase', self.data['id'], fieldname='showcase_id',
                                                           action=self.actions()['list_datasets'])
        datasets = list()
        if assoc_result:
            for dataset_dict in datasets_dicts:
                dataset = hdx.data.dataset.Dataset(dataset_dict, configuration=self.configuration)
                datasets.append(dataset)
        return datasets