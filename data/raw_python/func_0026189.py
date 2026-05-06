def _unquote(self, value):
        """Return an unquoted version of a value"""
        if not value:
            # should only happen during parsing of lists
            raise SyntaxError
        if (value[0] == value[-1]) and (value[0] in ('"', "'")):
            value = value[1:-1]
        return value