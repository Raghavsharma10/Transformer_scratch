def from_str(cls, s):
        # type: (Union[Text, bytes]) -> FmtStr
        r"""
        Return a FmtStr representing input.

        The str() of a FmtStr is guaranteed to produced the same FmtStr.
        Other input with escape sequences may not be preserved.

        >>> fmtstr("|"+fmtstr("hey", fg='red', bg='blue')+"|")
        '|'+on_blue(red('hey'))+'|'
        >>> fmtstr('|\x1b[31m\x1b[44mhey\x1b[49m\x1b[39m|')
        '|'+on_blue(red('hey'))+'|'
        """

        if '\x1b[' in s:
            try:
                tokens_and_strings = parse(s)
            except ValueError:
                return FmtStr(Chunk(remove_ansi(s)))
            else:
                chunks = []
                cur_fmt = {}
                for x in tokens_and_strings:
                    if isinstance(x, dict):
                        cur_fmt.update(x)
                    elif isinstance(x, (bytes, unicode)):
                        atts = parse_args('', dict((k, v)
                                          for k, v in cur_fmt.items()
                                          if v is not None))
                        chunks.append(Chunk(x, atts=atts))
                    else:
                        raise Exception("logic error")
                return FmtStr(*chunks)
        else:
            return FmtStr(Chunk(s))