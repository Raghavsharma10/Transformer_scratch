def _build_saveframe(self, lexer):
        """Build NMR-STAR file saveframe.

        :param lexer: instance of the lexical analyzer.
        :type lexer: :func:`~nmrstarlib.bmrblex.bmrblex`
        :return: Saveframe dictionary.
        :rtype: :py:class:`collections.OrderedDict`
        """
        odict = OrderedDict()
        loop_count = 0
        token = next(lexer)

        while token != u"save_":
            try:
                if token[0] == u"_":
                    # This strips off the leading underscore of tagnames for readability
                    odict[token[1:]] = next(lexer)

                    # Skip the saveframe if it's not in the list of wanted categories
                    if self._frame_categories:
                        if token == "_Saveframe_category" and odict[token[1:]] not in self._frame_categories:
                            raise SkipSaveFrame()

                elif token == u"loop_":
                    odict[u"loop_{}".format(loop_count)] = self._build_loop(lexer)
                    loop_count += 1

                elif token.lstrip().startswith(u"#"):
                    continue

                else:
                    print("Error: Invalid token {}".format(token), file=sys.stderr)
                    print("In _build_saveframe try block", file=sys.stderr)
                    raise InvalidToken("{}".format(token))

            except IndexError:
                print("Error: Invalid token {}".format(token), file=sys.stderr)
                print("In _build_saveframe except block", file=sys.stderr)
                raise
            except SkipSaveFrame:
                self._skip_saveframe(lexer)
                odict = None
            finally:
                if odict is None:
                    token = u"save_"
                else:
                    token = next(lexer)
        return odict