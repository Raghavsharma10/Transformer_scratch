def tokenize(self, text):
        """Tokenize given string `text`, and return a list of tokens. Raises
        :class:`~textparser.TokenizeError` on failure.

        This method should only be called by
        :func:`~textparser.Parser.parse()`, but may very well be
        overridden if the default implementation does not match the
        parser needs.

        """

        names, specs = self._unpack_token_specs()
        keywords = self.keywords()
        tokens, re_token = tokenize_init(specs)

        for mo in re.finditer(re_token, text, re.DOTALL):
            kind = mo.lastgroup

            if kind == 'SKIP':
                pass
            elif kind != 'MISMATCH':
                value = mo.group(kind)

                if value in keywords:
                    kind = value

                if kind in names:
                    kind = names[kind]

                tokens.append(Token(kind, value, mo.start()))
            else:
                raise TokenizeError(text, mo.start())

        return tokens