def get_tabular_rows(self, url, dict_rows=False, **kwargs):
        # type: (str, bool, Any) -> Iterator[Dict]
        """Get iterator for reading rows from tabular data. Each row is returned as a dictionary.

        Args:
            url (str): URL to download
            dict_rows (bool): Return dict (requires headers parameter) or list for each row. Defaults to False (list).
            **kwargs:
            headers (Union[int, List[int], List[str]]): Number of row(s) containing headers or list of headers
            file_type (Optional[str]): Type of file. Defaults to inferring.
            delimiter (Optional[str]): Delimiter used for values in each row. Defaults to inferring.

        Returns:
            Iterator[Union[List,Dict]]: Iterator where each row is returned as a list or dictionary.

        """
        return self.get_tabular_stream(url, **kwargs).iter(keyed=dict_rows)