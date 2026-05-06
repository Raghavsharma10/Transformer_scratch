def write(self, fname=None, filtered=False, header=True, append=False):
        r"""
        Write (processed) data to a specified comma-separated values (CSV) file.

        :param fname: Name of the comma-separated values file to be
                      written. If None the file from which the data originated
                      is overwritten
        :type  fname: FileName_

        :param filtered: Filtering type
        :type  filtered: :ref:`CsvFiltered`

        :param header: If a list, column headers to use in the file. If
                       boolean, flag that indicates whether the input column
                       headers should be written (True) or not (False)
        :type  header: string, list of strings or boolean

        :param append: Flag that indicates whether data is added to an
                       existing file (or a new file is created if it does not
                       exist) (True), or whether data overwrites the file
                       contents (if the file exists) or creates a new file if
                       the file does not exists (False)
        :type  append: boolean

        .. [[[cog cog.out(exobj.get_sphinx_autodoc()) ]]]
        .. Auto-generated exceptions documentation for
        .. pcsv.csv_file.CsvFile.write

        :raises:
         * OSError (File *[fname]* could not be created: *[reason]*)

         * RuntimeError (Argument \`append\` is not valid)

         * RuntimeError (Argument \`filtered\` is not valid)

         * RuntimeError (Argument \`fname\` is not valid)

         * RuntimeError (Argument \`header\` is not valid)

         * RuntimeError (Argument \`no_empty\` is not valid)

         * ValueError (There is no data to save to file)

        .. [[[end]]]
        """
        # pylint: disable=R0913
        write_ex = pexdoc.exh.addex(ValueError, "There is no data to save to file")
        fname = self._fname if fname is None else fname
        data = self.data(filtered=filtered)
        write_ex((len(data) == 0) or ((len(data) == 1) and (len(data[0]) == 0)))
        if header:
            header = [header] if isinstance(header, str) else header
            cfilter = self._gen_col_index(filtered=filtered)
            filtered_header = (
                [self._header[item] for item in cfilter]
                if self._has_header
                else cfilter
            )
            file_header = filtered_header if isinstance(header, bool) else header
        # Convert None's to ''
        data = [["''" if item is None else item for item in row] for row in data]
        _write_int(fname, [file_header] + data if header else data, append=append)