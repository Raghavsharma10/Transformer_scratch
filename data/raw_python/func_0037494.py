def print_saveframe(self, sf, f=sys.stdout, file_format="nmrstar", tw=3):
        """Print saveframe into a file or stdout.
        We need to keep track of how far over everything is tabbed. The "tab width"
        variable tw does this for us.

        :param str sf: Saveframe name.
        :param io.StringIO f: writable file-like stream.
        :param str file_format: Format to use: `nmrstar` or `json`.
        :param int tw: Tab width.
        :return: None
        :rtype: :py:obj:`None`
        """
        if file_format == "nmrstar":
            for sftag in self[sf].keys():
                # handle loops
                if sftag[:5] == "loop_":
                    print(u"\n{}loop_".format(tw * u" "), file=f)
                    self.print_loop(sf, sftag, f, file_format, tw * 2)
                    print(u"\n{}stop_".format(tw * u" "), file=f)

                # handle the NMR-Star "multiline string"
                elif self[sf][sftag].endswith(u"\n"):
                    print(u"{}_{}".format(tw * u" ", sftag), file=f)
                    print(u";\n{};".format(self[sf][sftag]), file=f)

                elif len(self[sf][sftag].split()) > 1:
                    # need to escape value with quotes (i.e. u"'{}'".format()) if value consists of two or more words
                    print(u"{}_{}\t {}".format(tw * u" ", sftag, u"'{}'".format(self[sf][sftag])), file=f)

                else:
                    print(u"{}_{}\t {}".format(tw * u" ", sftag, self[sf][sftag]), file=f)

        elif file_format == "json":
            print(json.dumps(self[sf], sort_keys=False, indent=4), file=f)