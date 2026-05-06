def parse(self, parser):
        '''parse content of extension'''
        # line number of token that started the tag
        lineno = next(parser.stream).lineno

        # template context
        context = nodes.ContextReference()

        # parse keyword arguments
        kwargs = []

        while parser.stream.look().type == lexer.TOKEN_ASSIGN:
            key = parser.stream.expect(lexer.TOKEN_NAME)
            next(parser.stream)
            kwargs.append(
                nodes.Keyword(key.value, parser.parse_expression()),
            )
            parser.stream.skip_if('comma')
        # parse content of the activeurl block up to endactiveurl
        body = parser.parse_statements(['name:endactiveurl'], drop_needle=True)

        args = [context]

        call_method = self.call_method(
            'render_tag',
            args=args,
            kwargs=kwargs,
        )

        return nodes.CallBlock(call_method, [], [], body).set_lineno(lineno)