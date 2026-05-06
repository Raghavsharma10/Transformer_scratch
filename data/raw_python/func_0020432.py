def load(self):
        """
        Extract tabular data as |TableData| instances from a CSV file.
        |load_source_desc_file|

        :return:
            Loaded table data.
            |load_table_name_desc|

            ===================  ========================================
            Format specifier     Value after the replacement
            ===================  ========================================
            ``%(filename)s``     |filename_desc|
            ``%(format_name)s``  ``"csv"``
            ``%(format_id)s``    |format_id_desc|
            ``%(global_id)s``    |global_id|
            ===================  ========================================
        :rtype: |TableData| iterator
        :raises pytablereader.DataError:
            If the CSV data is invalid.

        .. seealso::
            :py:func:`csv.reader`
        """

        self._validate()
        self._logger.logging_load()
        self.encoding = get_file_encoding(self.source, self.encoding)

        if six.PY3:
            self._csv_reader = csv.reader(
                io.open(self.source, "r", encoding=self.encoding),
                delimiter=self.delimiter,
                quotechar=self.quotechar,
                strict=True,
                skipinitialspace=True,
            )
        else:
            self._csv_reader = csv.reader(
                _utf_8_encoder(io.open(self.source, "r", encoding=self.encoding)),
                delimiter=self.delimiter,
                quotechar=self.quotechar,
                strict=True,
                skipinitialspace=True,
            )

        formatter = CsvTableFormatter(self._to_data_matrix())
        formatter.accept(self)

        return formatter.to_table_data()