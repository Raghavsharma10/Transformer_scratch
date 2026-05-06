def set_dataset_year_range(self, dataset_year, dataset_end_year=None):
        # type: (Union[str, int], Optional[Union[str, int]]) -> None
        """Set dataset date as a range from year or start and end year.

        Args:
            dataset_year (Union[str, int]): Dataset year given as string or int
            dataset_end_year (Optional[Union[str, int]]): Dataset end year given as string or int

        Returns:
            None
        """
        if isinstance(dataset_year, int):
            dataset_date = '01/01/%d' % dataset_year
        elif isinstance(dataset_year, str):
            dataset_date = '01/01/%s' % dataset_year
        else:
            raise hdx.data.hdxobject.HDXError('dataset_year has type %s which is not supported!' % type(dataset_year).__name__)
        if dataset_end_year is None:
            dataset_end_year = dataset_year
        if isinstance(dataset_end_year, int):
            dataset_end_date = '31/12/%d' % dataset_end_year
        elif isinstance(dataset_end_year, str):
            dataset_end_date = '31/12/%s' % dataset_end_year
        else:
            raise hdx.data.hdxobject.HDXError('dataset_end_year has type %s which is not supported!' % type(dataset_end_year).__name__)
        self.set_dataset_date(dataset_date, dataset_end_date)