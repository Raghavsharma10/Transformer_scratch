def get(key, default=None):
    """
        Searches os.environ. If a key is found try evaluating its type else;
        return the string.

        returns: k->value (type as defined by ast.literal_eval)
    """
    try:
        # Attempt to evaluate into python literal
        return ast.literal_eval(os.environ.get(key.upper(), default))
    except (ValueError, SyntaxError):
        return os.environ.get(key.upper(), default)