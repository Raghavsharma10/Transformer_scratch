def print_file(self, f=sys.stdout, file_format="cif", tw=0):
        """Print :class:`~nmrstarlib.nmrstarlib.CIFFile` into a file or stdout.

        :param io.StringIO f: writable file-like stream.
        :param str file_format: Format to use: `cif` or `json`.
        :param int tw: Tab width.
        :return: None
        :rtype: :py:obj:`None`
        """
        if file_format == "cif":
            for key in self.keys():
                if key == u"data":
                    print(u"{}_{}".format(key, self[key]), file=f)
                elif key.startswith(u"comment"):
                    print(u"{}".format(self[key].strip()), file=f)
                elif key.startswith(u"loop_"):
                    print(u"{}loop_".format(tw * u" "), file=f)
                    self.print_loop(key, f, file_format, tw)
                else:
                    # handle the NMR-Star "multiline string"
                    if self[key].endswith(u"\n"):
                        print(u"{}_{}".format(tw * u" ", key), file=f)
                        print(u";{};".format(self[key]), file=f)

                    # need to escape value with quotes (i.e. u"'{}'".format()) if value consists of two or more words
                    elif len(self[key].split()) > 1:
                        print(u"{}_{}\t {}".format(tw * u" ", key, u"'{}'".format(self[key])), file=f)

                    else:
                        print(u"{}_{}\t {}".format(tw * u" ", key, self[key]), file=f)

        elif file_format == "json":
            print(self._to_json(), file=f)