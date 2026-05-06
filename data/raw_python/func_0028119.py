def set_dataset_date_from_datetime(self, dataset_date, dataset_end_date=None):
        # type: (datetime, Optional[datetime]) -> None
        """Set dataset date from datetime.datetime object

        Args:
            dataset_date (datetime.datetime): Dataset date
            dataset_end_date (Optional[datetime.datetime]): Dataset end date

        Returns:
            None
        """
        start_date = dataset_date.strftime('%m/%d/%Y')
        if dataset_end_date is None:
            self.data['dataset_date'] = start_date
        else:
            end_date = dataset_end_date.strftime('%m/%d/%Y')
            self.data['dataset_date'] = '%s-%s' % (start_date, end_date)