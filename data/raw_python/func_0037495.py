def print_loop(self, sf, sftag, f=sys.stdout, file_format="nmrstar", tw=3):
        """Print loop into a file or stdout.

        :param str sf: Saveframe name.
        :param str sftag: Saveframe tag, i.e. field name.
        :param io.StringIO f: writable file-like stream.
        :param str file_format: Format to use: `nmrstar` or `json`.
        :param int tw: Tab width.
        :return: None
        :rtype: :py:obj:`None`
        """
        if file_format == "nmrstar":
            # First print the fields
            for field in self[sf][sftag][0]:
                print(u"{}_{}".format(tw * u" ", field), file=f)

            print(u"", file=f)  # new line between fields and values

            # Then print the values
            for valuesdict in self[sf][sftag][1]:
                # need to escape value with quotes (i.e. u"'{}'".format()) if value consists of two or more words
                print(u"{}{}".format(tw * u" ", u" ".join([u"'{}'".format(value) if len(value.split()) > 1 else value for value
                                                           in valuesdict.values()])), file=f)
        elif file_format == "json":
            print(json.dumps(self[sf][sftag], sort_keys=False, indent=4), file=f)