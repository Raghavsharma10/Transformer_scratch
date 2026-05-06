def _tokenize(self, text, token_class=None):
        """
        Tokenizes a text

        :Returns:
            A `list` of tokens
        """
        token_class = token_class or Token
        tokens = {}

        for i, match in enumerate(self.regex.finditer(text)):
            value = match.group(0)

            try:
                token = tokens[value]
            except KeyError:
                type = match.lastgroup
                token = token_class(value, type=type)
                tokens[value] = token

            yield token