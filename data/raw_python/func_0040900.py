def call(self, tokens, *args, **kwargs):
        """Add args and kwargs to the tokens.
        """
        tokens.append([evaluate, [args, kwargs], {}])
        return tokens