def parser(*events):
    """Decorator for convenience - adds a function as a parser for event(s)."""
    def dec(func):
        for event in events:
            PARSERS[event] = func
        return func
    return dec