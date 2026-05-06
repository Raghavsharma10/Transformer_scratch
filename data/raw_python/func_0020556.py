def load(self):
        """
        Extract tabular data as |TableData| instances from a MediaWiki file.
        |load_source_desc_file|

        :return:
            Loaded table data iterator.
            |load_table_name_desc|

            ===================  ==============================================
            Format specifier     Value after the replacement
            ===================  ==============================================
            ``%(filename)s``     |filename_desc|
            ``%(key)s``          | This replaced to:
                                 | **(1)** ``caption`` mark of the table
                                 | **(2)** ``%(format_name)s%(format_id)s``
                                 | if ``caption`` mark not included
                                 | in the table.
            ``%(format_name)s``  ``"mediawiki"``
            ``%(format_id)s``    |format_id_desc|
            ``%(global_id)s``    |global_id|
            ===================  ==============================================
        :rtype: |TableData| iterator
        :raises pytablereader.DataError:
            If the MediaWiki data is invalid or empty.
        """

        self._validate()
        self._logger.logging_load()
        self.encoding = get_file_encoding(self.source, self.encoding)

        with io.open(self.source, "r", encoding=self.encoding) as fp:
            formatter = MediaWikiTableFormatter(fp.read())
        formatter.accept(self)

        return formatter.to_table_data()