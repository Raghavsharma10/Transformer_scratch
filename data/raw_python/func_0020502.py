def load(self):
        """
        Load table data from a Google Spreadsheet.

        This method consider :py:attr:`.source` as a path to the
        credential JSON file to access Google Sheets API.

        The method automatically search the header row start from
        :py:attr:`.start_row`. The condition of the header row is that
        all of the columns have value (except empty columns).

        :return:
            Loaded table data. Return one |TableData| for each sheet in
            the workbook. The table name for data will be determined by
            :py:meth:`~.GoogleSheetsTableLoader.make_table_name`.
        :rtype: iterator of |TableData|
        :raises pytablereader.DataError:
            If the header row is not found.
        :raises pytablereader.OpenError:
            If the spread sheet not found.
        """

        import gspread
        from oauth2client.service_account import ServiceAccountCredentials

        self._validate_table_name()
        self._validate_title()

        scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
        credentials = ServiceAccountCredentials.from_json_keyfile_name(self.source, scope)

        gc = gspread.authorize(credentials)
        try:
            for worksheet in gc.open(self.title).worksheets():
                self._worksheet = worksheet
                self.__all_values = [row for row in worksheet.get_all_values()]

                if self._is_empty_sheet():
                    continue

                try:
                    self.__strip_empty_col()
                except ValueError:
                    continue

                value_matrix = self.__all_values[self._get_start_row_idx() :]
                try:
                    headers = value_matrix[0]
                    rows = value_matrix[1:]
                except IndexError:
                    continue

                self.inc_table_count()

                yield TableData(
                    self.make_table_name(),
                    headers,
                    rows,
                    dp_extractor=self.dp_extractor,
                    type_hints=self._extract_type_hints(headers),
                )
        except gspread.exceptions.SpreadsheetNotFound:
            raise OpenError("spreadsheet '{}' not found".format(self.title))
        except gspread.exceptions.APIError as e:
            raise APIError(e)