def _build_file(self, nmrstar_str):
        """Build :class:`~nmrstarlib.nmrstarlib.NMRStarFile` object.

        :param nmrstar_str: NMR-STAR-formatted string.
        :type nmrstar_str: :py:class:`str` or :py:class:`bytes`
        :return: instance of :class:`~nmrstarlib.nmrstarlib.NMRStarFile`.
        :rtype: :class:`~nmrstarlib.nmrstarlib.NMRStarFile`
        """
        odict = self
        comment_count = 0
        lexer = bmrblex(nmrstar_str)
        token = next(lexer)

        while token != u"":
            try:
                if token[0:5] == u"save_":
                    name = token
                    frame = self._build_saveframe(lexer)
                    if frame:
                        odict[name] = frame

                elif token[0:5] == u"data_":
                    self.id = token[5:]
                    odict[u"data"] = self.id

                elif token.lstrip().startswith(u"#"):
                    odict[u"comment_{}".format(comment_count)] = token
                    comment_count += 1

                else:
                    print("Error: Invalid token {}".format(token), file=sys.stderr)
                    print("In _build_starfile try block", file=sys.stderr)
                    raise InvalidToken("{}".format(token))

            except IndexError:
                print("Error: Invalid token {}".format(token), file=sys.stderr)
                print("In _build_starfile except block", file=sys.stderr)
                raise

            finally:
                token = next(lexer)
        return self