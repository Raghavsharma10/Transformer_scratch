def to_csv(self, filename=None, *, fields=None, append=False, header=True, header_prefix='', sep=',', newline='\n'):
        """
        Parameters
        ----------
        filename: str or None
            The file to which output will be written. By default, any existing content is
            overwritten. Use `append=True` to open the file in append mode instead.
            If filename is None, the generated CSV output is returned instead of written
            to a file.
        fields: list or dict
            List of field names to export, or dictionary mapping output column names
            to attribute names of the generators.

            Examples:
               fields=['field_name_1', 'field_name_2']
               fields={'COL1': 'field_name_1', 'COL2': 'field_name_2'}
        append: bool
            If `True`, open the file in 'append' mode to avoid overwriting existing content.
            Default is `False`, i.e. any existing content will be overwritten.
            This argument only has an effect if `filename` is given (i.e. if output happens
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
        The return value depends on the value of `filename`.
        If `filename` is given, writes the output to the file and returns `None`.
        If `filename` is `None`, returns a string containing the CSV output.
        """
        assert isinstance(append, bool)

        if fields is None:
            raise NotImplementedError("TODO: derive field names automatically from the generator which produced this item list")

        if isinstance(fields, (list, tuple)):
            fields = {name: name for name in fields}

        header_line = _generate_csv_header_line(header=header, header_prefix=header_prefix, header_names=fields.keys(), sep=sep, newline=newline)

        if filename is not None:
            # ensure parent directory of output file exits
            dirname = os.path.dirname(os.path.abspath(filename))
            if not os.path.exists(dirname):
                os.makedirs(dirname)
        file_or_string = open(filename, 'a' if append else 'w') if (filename is not None) else io.StringIO()

        retval = None
        attr_getters = [attrgetter(attr_name) for attr_name in fields.values()]
        try:

            file_or_string.write(header_line)

            for x in self.items:
                line = sep.join([format(func(x)) for func in attr_getters]) + newline
                file_or_string.write(line)

            if filename is None:
                retval = file_or_string.getvalue()

        finally:
            file_or_string.close()

        return retval