def download_tabular_rows_as_dicts(self, url, headers=1, keycolumn=1, **kwargs):
        # type: (str, Union[int, List[int], List[str]], int, Any) -> Dict[Dict]
        """Download multicolumn csv from url and return dictionary where keys are first column and values are
        dictionaries with keys from column headers and values from columns beneath

        Args:
            url (str): URL to download
            headers (Union[int, List[int], List[str]]): Number of row(s) containing headers or list of headers. Defaults to 1.
            keycolumn (int): Number of column to be used for key. Defaults to 1.
            **kwargs:
            file_type (Optional[str]): Type of file. Defaults to inferring.
            delimiter (Optional[str]): Delimiter used for values in each row. Defaults to inferring.

        Returns:
            Dict[Dict]: Dictionary where keys are first column and values are dictionaries with keys from column
            headers and values from columns beneath

        """
        kwargs['headers'] = headers
        stream = self.get_tabular_stream(url, **kwargs)
        output_dict = dict()
        headers = stream.headers
        key_header = headers[keycolumn - 1]
        for row in stream.iter(keyed=True):
            first_val = row[key_header]
            output_dict[first_val] = dict()
            for header in row:
                if header == key_header:
                    continue
                else:
                    output_dict[first_val][header] = row[header]
        return output_dict