def _write_packed_data(self, data_out, table):
        """This is kind of legacy function - this functionality may be useful for some people, so even though
        now the default of writing CSV is writing unpacked data (divided by independent variable) this method is
        still available and accessible if ```pack``` flag is specified in Writer's options

        :param output: output file like object to which data will be written
        :param table: input table
        :type table: hepdata_converter.parsers.Table
        """
        headers = []
        data = []
        qualifiers_marks = []
        qualifiers = {}

        self._extract_independent_variables(table, headers, data, qualifiers_marks)

        for dependent_variable in table.dependent_variables:
            self._parse_dependent_variable(dependent_variable, headers, qualifiers, qualifiers_marks, data)

        self._write_metadata(data_out, table)
        self._write_csv_data(data_out, qualifiers, qualifiers_marks, headers, data)