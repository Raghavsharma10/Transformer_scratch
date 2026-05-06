def _parse_date(dataset_date, date_format):
        # type: (str, Optional[str]) -> datetime
        """Parse dataset date from string using specified format. If no format is supplied, the function will guess.
        For unambiguous formats, this should be fine.

        Args:
            dataset_date (str): Dataset date string
            date_format (Optional[str]): Date format. If None is given, will attempt to guess. Defaults to None.

        Returns:
            datetime.datetime
        """
        if date_format is None:
            try:
                return parser.parse(dataset_date)
            except (ValueError, OverflowError) as e:
                raisefrom(HDXError, 'Invalid dataset date!', e)
        else:
            try:
                return datetime.strptime(dataset_date, date_format)
            except ValueError as e:
                raisefrom(HDXError, 'Invalid dataset date!', e)