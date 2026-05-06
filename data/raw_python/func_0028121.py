def set_dataset_date(self, dataset_date, dataset_end_date=None, date_format=None):
        # type: (str, Optional[str], Optional[str]) -> None
        """Set dataset date from string using specified format. If no format is supplied, the function will guess.
        For unambiguous formats, this should be fine.

        Args:
            dataset_date (str): Dataset date string
            dataset_end_date (Optional[str]): Dataset end date string
            date_format (Optional[str]): Date format. If None is given, will attempt to guess. Defaults to None.

        Returns:
            None
        """
        parsed_date = self._parse_date(dataset_date, date_format)
        if dataset_end_date is None:
            self.set_dataset_date_from_datetime(parsed_date)
        else:
            parsed_end_date = self._parse_date(dataset_end_date, date_format)
            self.set_dataset_date_from_datetime(parsed_date, parsed_end_date)