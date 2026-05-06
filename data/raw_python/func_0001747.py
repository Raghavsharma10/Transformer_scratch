def download_tabular_key_value(self, url, **kwargs):
        # type: (str, Any) -> Dict
        """Download 2 column csv from url and return a dictionary of keys (first column) and values (second column)

        Args:
            url (str): URL to download
            **kwargs:
            headers (Union[int, List[int], List[str]]): Number of row(s) containing headers or list of headers
            file_type (Optional[str]): Type of file. Defaults to inferring.
            delimiter (Optional[str]): Delimiter used for values in each row. Defaults to inferring.

        Returns:
            Dict: Dictionary keys (first column) and values (second column)

        """
        output_dict = dict()
        for row in self.get_tabular_rows(url, **kwargs):
            if len(row) < 2:
                continue
            output_dict[row[0]] = row[1]
        return output_dict