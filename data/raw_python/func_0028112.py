def get_all_datasets(cls, configuration=None, page_size=1000, check_duplicates=True, **kwargs):
        # type: (Optional[Configuration], int, bool, Any) -> List['Dataset']
        """Get all datasets in HDX

        Args:
            configuration (Optional[Configuration]): HDX configuration. Defaults to global configuration.
            page_size (int): Size of page to return. Defaults to 1000.
            check_duplicates (bool): Whether to check for duplicate datasets. Defaults to True.
            **kwargs: See below
            limit (int): Number of rows to return. Defaults to all datasets (sys.maxsize)
            offset (int): Offset in the complete result for where the set of returned datasets should begin

        Returns:
            List[Dataset]: list of all datasets in HDX
        """

        dataset = Dataset(configuration=configuration)
        dataset['id'] = 'all datasets'  # only for error message if produced
        total_rows = kwargs.get('limit', cls.max_int)
        start = kwargs.get('offset', 0)
        all_datasets = None
        attempts = 0
        while attempts < cls.max_attempts and all_datasets is None:  # if the dataset names vary for multiple calls, then must redo query
            all_datasets = list()
            for page in range(total_rows // page_size + 1):
                pagetimespagesize = page * page_size
                kwargs['offset'] = start + pagetimespagesize
                rows_left = total_rows - pagetimespagesize
                rows = min(rows_left, page_size)
                kwargs['limit'] = rows
                result = dataset._write_to_hdx('all', kwargs, 'id')
                datasets = list()
                if isinstance(result, list):
                    no_results = len(result)
                    if no_results == 0 and page == 0:
                        all_datasets = None
                        break
                    for datasetdict in result:
                        dataset = Dataset(configuration=configuration)
                        dataset.old_data = dict()
                        dataset.data = datasetdict
                        dataset._dataset_create_resources()
                        datasets.append(dataset)
                    all_datasets += datasets
                    if no_results < rows:
                        break
                else:
                    logger.debug(result)
            if all_datasets is None:
                attempts += 1
            elif check_duplicates:
                names_list = [dataset['name'] for dataset in all_datasets]
                names = set(names_list)
                if len(names_list) != len(names):  # check for duplicates (shouldn't happen)
                    all_datasets = None
                    attempts += 1
                # This check is no longer valid because of showcases being returned by package_list!
                # elif total_rows == max_int:
                #     all_names = set(Dataset.get_all_dataset_names())  # check dataset names match package_list
                #     if names != all_names:
                #         all_datasets = None
                #         attempts += 1
        if attempts == cls.max_attempts and all_datasets is None:
            raise HDXError('Maximum attempts reached for getting all datasets!')
        return all_datasets