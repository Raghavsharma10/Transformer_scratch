def _wrap_parse(code, filename):
        """
        async wrapper is required to avoid await calls raising a SyntaxError
        """
        code = 'async def wrapper():\n' + indent(code, ' ')
        return ast.parse(code, filename=filename).body[0].body[0].value