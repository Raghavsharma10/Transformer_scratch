def _build_loop(self, lexer):
        """Build saveframe loop.

        :param lexer: instance of lexical analyzer.
        :type lexer: :func:`~nmrstarlib.bmrblex.bmrblex`
        :return: Fields and values of the loop.
        :rtype: :py:class:`tuple`
        """
        fields = []
        values = []

        token = next(lexer)
        while token[0] == u"_":
            fields.append(token[1:])
            token = next(lexer)

        while token != u"stop_":
            values.append(token)
            token = next(lexer)

        assert float(len(values) / len(fields)).is_integer(), \
            "Error in loop construction: number of fields must be equal to number of values."

        values = [OrderedDict(zip(fields, values[i:i + len(fields)])) for i in range(0, len(values), len(fields))]
        return fields, values