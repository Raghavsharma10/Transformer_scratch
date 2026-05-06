def _build_file(self, cif_str):
        """Build :class:`~nmrstarlib.nmrstarlib.CIFFile` object.

        :param cif_str: NMR-STAR-formatted string.
        :type cif_str: :py:class:`str` or :py:class:`bytes`
        :return: instance of :class:`~nmrstarlib.nmrstarlib.CIFFile`.
        :rtype: :class:`~nmrstarlib.nmrstarlib.CIFFile`
        """
        odict = self
        comment_count = 0
        loop_count = 0
        lexer = bmrblex(cif_str)
        token = next(lexer)

        while token != u"":
            try:
                if token[0:5] == u"data_":
                    self.id = token[5:]
                    self[u"data"] = self.id

                elif token.lstrip().startswith(u"#"):
                    odict[u"comment_{}".format(comment_count)] = token
                    comment_count += 1

                elif token[0] == u"_":
                    # This strips off the leading underscore of tagnames for readability
                    value = next(lexer)
                    odict[token[1:]] = value

                elif token == u"loop_":
                    odict[u"loop_{}".format(loop_count)] = self._build_loop(lexer)
                    loop_count += 1

                else:
                    print("Error: Invalid token {}".format(token), file=sys.stderr)
                    print("In _build_file try block", file=sys.stderr)
                    raise InvalidToken("{}".format(token))

            except IndexError:
                print("Error: Invalid token {}".format(token), file=sys.stderr)
                print("In _build_file except block", file=sys.stderr)
                raise

            finally:
                token = next(lexer)
        return self