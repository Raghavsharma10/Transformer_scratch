def emit(self, string, match, pattern, **_):
        """Emits a token using the current pattern match and pattern label."""
        return grammar.Token(name=pattern.name, value=string,
                             start=match.start(), end=match.end())