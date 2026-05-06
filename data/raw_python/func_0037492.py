def _skip_saveframe(self, lexer):
        """Skip entire saveframe - keep emitting tokens until the end of saveframe.

        :param lexer: instance of the lexical analyzer class.
        :type lexer: :class:`~nmrstarlib.bmrblex.bmrblex`
        :return: None
        :rtype: :py:obj:`None`
        """
        token = u""
        while token != u"save_":
            token = next(lexer)