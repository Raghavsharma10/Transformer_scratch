def search_in_hdx(cls, query='*:*', configuration=None, page_size=1000, **kwargs):
        # type: (Optional[str], Optional[Configuration], int, Any) -> List['Dataset']
        """Searches for datasets in HDX

        Args:
            query (Optional[str]): Query (in Solr format). Defaults to '*:*'.
            configuration (Optional[Configuration]): HDX configuration. Defaults to global configuration.
            page_size (int): Size of page to return. Defaults to 1000.
            **kwargs: See below
            fq (string): Any filter queries to apply
            sort (string): Sorting of the search results. Defaults to 'relevance asc, metadata_modified desc'.
            rows (int): Number of matching rows to return. Defaults to all datasets (sys.maxsize).
            start (int): Offset in the complete result for where the set of returned datasets should begin
            facet (string): Whether to enable faceted results. Default to True.
            facet.mincount (int): Minimum counts for facet fields should be included in the results
            facet.limit (int): Maximum number of values the facet fields return (- = unlimited). Defaults to 50.
            facet.field (List[str]): Fields to facet upon. Default is empty.
            use_default_schema (bool): Use default package schema instead of custom schema. Defaults to False.

        Returns:
            List[Dataset]: list of datasets resulting from query
        """

        dataset = Dataset(configuration=configuration)
        total_rows = kwargs.get('rows', cls.max_int)
        start = kwargs.get('start', 0)
        all_datasets = None
        attempts = 0
        while attempts < cls.max_attempts and all_datasets is None:  # if the count values vary for multiple calls, then must redo query
            all_datasets = list()
            counts = set()
            for page in range(total_rows // page_size + 1):
                pagetimespagesize = page * page_size
                kwargs['start'] = start + pagetimespagesize
                rows_left = total_rows - pagetimespagesize
                rows = min(rows_left, page_size)
                kwargs['rows'] = rows
                _, result = dataset._read_from_hdx('dataset', query, 'q', Dataset.actions()['search'], **kwargs)
                datasets = list()
                if result:
                    count = result.get('count', None)
                    if count:
                        counts.add(count)
                        no_results = len(result['results'])
                        for datasetdict in result['results']:
                            dataset = Dataset(configuration=configuration)
                            dataset.old_data = dict()
                            dataset.data = datasetdict
                            dataset._dataset_create_resources()
                            datasets.append(dataset)
                        all_datasets += datasets
                        if no_results < rows:
                            break
                    else:
                        break
                else:
                    logger.debug(result)
            if all_datasets and len(counts) != 1:  # Make sure counts are all same for multiple calls to HDX
                all_datasets = None
                attempts += 1
            else:
                ids = [dataset['id'] for dataset in all_datasets]  # check for duplicates (shouldn't happen)
                if len(ids) != len(set(ids)):
                    all_datasets = None
                    attempts += 1
        if attempts == cls.max_attempts and all_datasets is None:
            raise HDXError('Maximum attempts reached for searching for datasets!')
        return all_datasets