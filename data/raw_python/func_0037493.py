def print_file(self, f=sys.stdout, file_format="nmrstar", tw=3):
        """Print :class:`~nmrstarlib.nmrstarlib.NMRStarFile` into a file or stdout.

        :param io.StringIO f: writable file-like stream.
        :param str file_format: Format to use: `nmrstar` or `json`.
        :param int tw: Tab width.
        :return: None
        :rtype: :py:obj:`None`
        """
        if file_format == "nmrstar":
            for saveframe in self.keys():
                if saveframe == u"data":
                    print(u"{}_{}\n".format(saveframe, self[saveframe]), file=f)
                elif saveframe.startswith(u"comment"):
                    print(u"{}".format(self[saveframe]), file=f)
                else:
                    print(u"{}".format(saveframe), file=f)
                    self.print_saveframe(saveframe, f, file_format, tw)
                    print(u"\nsave_\n\n", file=f)

        elif file_format == "json":
            print(self._to_json(), file=f)