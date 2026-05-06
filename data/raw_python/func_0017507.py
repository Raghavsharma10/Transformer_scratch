def parse(self, text, token_tree=False, match_sof=False):
        """Parse given string `text` and return the parse tree. Raises
        :class:`~textparser.ParseError` on failure.

        Returns a parse tree of tokens if `token_tree` is ``True``.

        .. code-block:: python

           >>> MyParser().parse('Hello, World!')
           ['Hello', ',', 'World', '!']
           >>> tree = MyParser().parse('Hello, World!', token_tree=True)
           >>> from pprint import pprint
           >>> pprint(tree)
           [Token(kind='WORD', value='Hello', offset=0),
            Token(kind=',', value=',', offset=5),
            Token(kind='WORD', value='World', offset=7),
            Token(kind='!', value='!', offset=12)]

        """

        try:
            tokens = self.tokenize(text)

            if len(tokens) == 0 or tokens[-1].kind != '__EOF__':
                tokens.append(Token('__EOF__', '__EOF__', len(text)))

            if not match_sof:
                if len(tokens) > 0 and tokens[0].kind == '__SOF__':
                    del tokens[0]

            return Grammar(self.grammar()).parse(tokens, token_tree)
        except (TokenizeError, GrammarError) as e:
            raise ParseError(text, e.offset)