def _get_formatted_date(dataset_date, date_format=None):
        # type: (Optional[datetime], Optional[str]) -> Optional[str]
        """Get supplied dataset date as string in specified format. 
        If no format is supplied, an ISO 8601 string is returned.

        Args:
            dataset_date (Optional[datetime.datetime]): dataset date in datetime.datetime format 
            date_format (Optional[str]): Date format. None is taken to be ISO 8601. Defaults to None.

        Returns:
            Optional[str]: Dataset date string or None if no date is set
        """
        if dataset_date:
            if date_format:
                return dataset_date.strftime(date_format)
            else:
                return dataset_date.date().isoformat()
        else:
            return None