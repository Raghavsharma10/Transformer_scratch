def load(self):
        """
        Extract tabular data as |TableData| instances from a LTSV text object.
        |load_source_desc_text|

        :return:
            Loaded table data.
            |load_table_name_desc|

            ===================  ========================================
            Format specifier     Value after the replacement
            ===================  ========================================
            ``%(filename)s``     ``""``
            ``%(format_name)s``  ``"ltsv"``
            ``%(format_id)s``    |format_id_desc|
            ``%(global_id)s``    |global_id|
            ===================  ========================================
        :rtype: |TableData| iterator
        :raises pytablereader.InvalidHeaderNameError:
            If an invalid label name is included in the LTSV file.
        :raises pytablereader.DataError:
            If the LTSV data is invalid.
        """

        self._validate()
        self._logger.logging_load()

        self._ltsv_input_stream = self.source.splitlines()

        for data_matrix in self._to_data_matrix():
            formatter = SingleJsonTableConverterA(data_matrix)
            formatter.accept(self)

            return formatter.to_table_data()