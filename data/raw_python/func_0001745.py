def get_tabular_stream(self, url, **kwargs):
        # type: (str, Any) -> tabulator.Stream
        """Get Tabulator stream.

        Args:
            url (str): URL to download
            **kwargs:
            headers (Union[int, List[int], List[str]]): Number of row(s) containing headers or list of headers
            file_type (Optional[str]): Type of file. Defaults to inferring.
            delimiter (Optional[str]): Delimiter used for values in each row. Defaults to inferring.

        Returns:
            tabulator.Stream: Tabulator Stream object

        """
        self.close_response()
        file_type = kwargs.get('file_type')
        if file_type is not None:
            kwargs['format'] = file_type
            del kwargs['file_type']
        try:
            self.response = tabulator.Stream(url, **kwargs)
            self.response.open()
            return self.response
        except TabulatorException as e:
            raisefrom(DownloadError, 'Getting tabular stream for %s failed!' % url, e)