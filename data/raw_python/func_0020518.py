def load(self):
        """
        Extract tabular data as |TableData| instances from a SQLite database
        file. |load_source_desc_file|

        :return:
            Loaded table data iterator.
            |load_table_name_desc|

            ===================  ==============================================
            Format specifier     Value after the replacement
            ===================  ==============================================
            ``%(filename)s``     |filename_desc|
            ``%(key)s``          ``%(format_name)s%(format_id)s``
            ``%(format_name)s``  ``"sqlite"``
            ``%(format_id)s``    |format_id_desc|
            ``%(global_id)s``    |global_id|
            ===================  ==============================================
        :rtype: |TableData| iterator
        :raises pytablereader.DataError:
            If the SQLite database file data is invalid or empty.
        """

        self._validate()

        formatter = SqliteTableFormatter(self.source)
        formatter.accept(self)

        return formatter.to_table_data()