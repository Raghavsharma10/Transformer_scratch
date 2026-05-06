def load(self):
        """
        Extract tabular data as |TableData| instances from an Excel file.
        |spreadsheet_load_desc|

        :return:
            Loaded |TableData| iterator.
            |TableData| created for each sheet in the workbook.
            |load_table_name_desc|

            ===================  ====================================
            Format specifier     Value after the replacement
            ===================  ====================================
            ``%(filename)s``     Filename of the workbook
            ``%(sheet)s``        Name of the sheet
            ``%(format_name)s``  ``"spreadsheet"``
            ``%(format_id)s``    |format_id_desc|
            ``%(global_id)s``    |global_id|
            ===================  ====================================
        :rtype: |TableData| iterator
        :raises pytablereader.DataError:
            If the header row is not found.
        :raises pytablereader.error.OpenError:
            If failed to open the source file.
        """

        import xlrd

        self._validate()
        self._logger.logging_load()

        try:
            workbook = xlrd.open_workbook(self.source)
        except xlrd.biffh.XLRDError as e:
            raise OpenError(e)

        for worksheet in workbook.sheets():
            self._worksheet = worksheet

            if self._is_empty_sheet():
                continue

            self.__extract_not_empty_col_idx()

            try:
                start_row_idx = self._get_start_row_idx()
            except DataError:
                continue

            rows = [
                self.__get_row_values(row_idx)
                for row_idx in range(start_row_idx + 1, self._row_count)
            ]

            self.inc_table_count()
            headers = self.__get_row_values(start_row_idx)

            yield TableData(
                self._make_table_name(),
                headers,
                rows,
                dp_extractor=self.dp_extractor,
                type_hints=self._extract_type_hints(headers),
            )