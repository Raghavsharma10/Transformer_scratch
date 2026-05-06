def to_csv(self, output_file=None, *, fields=None, fields_to_explode=None, append=False, header=True, header_prefix='', sep=',', newline='\n'):
        """
        Parameters
        ----------
        output_file: str or file object or None
            The file to which output will be written. By default, any existing content is
            overwritten. Use `append=True` to open the file in append mode instead.
            If `output_file` is None, the generated CSV output is returned as a string
            instead of written to a file.
        fields: list or dict
            List of field names to export, or dictionary mapping output column names
            to attribute names of the generators.

            Examples:
               fields=['field_name_1', 'field_name_2']
               fields={'COL1': 'field_name_1', 'COL2': 'field_name_2'}
        fields_to_explode: list
            Optional list of field names where each entry (which must itself be a sequence)
            is to be "exploded" into separate rows. (*Note:* this is not supported yet for CSV export.)
        append: bool
            If `True`, open the file in 'append' mode to avoid overwriting existing content.
            Default is `False`, i.e. any existing content will be overwritten.
            This argument only has an effect if `output_file` is given (i.e. if output happens
            to a file instead of returning a CSV string).
        header: bool or str or None
            If `header=False` or `header=None` then no header line will be written.
            If `header` is a string then this string will be used as the header line.
            If `header=True` then a header line will be automatically generated from
            the field names of the custom generator.
        header_prefix: str
            If `header=True` then the auto-generated header line will be prefixed
            with `header_prefix` (otherwise this argument has no effect). For example,
            set `header_prefix='#'` to make the header line start with '#'. Default: ''
        sep: str
            Field separator to use in the output. Default: ','
        newline: str
            Line terminator to use in the output. Default: '\n'

        Returns
        -------
        The return value depends on the value of `output_file`.
        If `output_file` is given, writes the output to the file and returns `None`.
        If `output_file` is `None`, returns a string containing the CSV output.
        """
        assert isinstance(append, bool)

        if fields is None:
            raise NotImplementedError("TODO: derive field names automatically from the generator which produced this item list")

        if fields_to_explode is not None:
            raise NotImplementedError("TODO: the 'fields_to_explode' argument is not supported for CSV export yet.")

        if isinstance(fields, (list, tuple)):
            fields = {name: name for name in fields}

        header_line = _generate_csv_header_line(header=header, header_prefix=header_prefix, header_names=fields.keys(), sep=sep, newline=newline)

        if output_file is None:
            file_or_string = io.StringIO()
        elif isinstance(output_file, str):
            mode = 'a' if append else 'w'
            file_or_string = open(output_file, mode)

            # ensure parent directory of output file exits
            dirname = os.path.dirname(os.path.abspath(output_file))
            if not os.path.exists(dirname):
                logger.debug(f"Creating parent directory of output file '{output_file}'")
                os.makedirs(dirname)

        elif isinstance(output_file, io.IOBase):
            file_or_string = output_file
        else:
            raise TypeError(f"Invalid output file: {output_file} (type: {type(output_file)})")

        retval = None
        attr_getters = [attrgetter(attr_name) for attr_name in fields.values()]
        try:
            # TODO: quick-and-dirty solution to enable writing to gzip files; tidy this up!
            # (Note that for regular file output we don't want to encode each line to a bytes
            # object because this seems to be ca. 2x slower).
            if isinstance(file_or_string, gzip.GzipFile):
                file_or_string.write(header_line.encode())
                for x in self.items:
                    line = sep.join([format(func(x)) for func in attr_getters]) + newline
                    file_or_string.write(line.encode())

            else:
                file_or_string.write(header_line)
                for x in self.items:
                    line = sep.join([format(func(x)) for func in attr_getters]) + newline
                    file_or_string.write(line)

            if output_file is None:
                retval = file_or_string.getvalue()

        finally:
            file_or_string.close()

        return retval