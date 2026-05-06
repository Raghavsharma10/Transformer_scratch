def load(self):
        """
        Extract tabular data as |TableData| instances from a CSV text object.
        |load_source_desc_text|

        :return:
            Loaded table data.
            |load_table_name_desc|

            ===================  ========================================
            Format specifier     Value after the replacement
            ===================  ========================================
            ``%(filename)s``     ``""``
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

        self._csv_reader = csv.reader(
            six.StringIO(self.source.strip()),
            delimiter=self.delimiter,
            quotechar=self.quotechar,
            strict=True,
            skipinitialspace=True,
        )
        formatter = CsvTableFormatter(self._to_data_matrix())
        formatter.accept(self)

        return formatter.to_table_data()