def parse(self, parser):
        """Main method to render data into the template."""
        lineno = next(parser.stream).lineno

        if parser.stream.skip_if('name:short'):
            parser.stream.skip(1)
            short = parser.parse_expression()
        else:
            short = nodes.Const(False)

        result = self.call_method('_commit_hash', [short], [], lineno=lineno)
        return nodes.Output([result], lineno=lineno)