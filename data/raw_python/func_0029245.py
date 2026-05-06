def issequence(arg):
    """Return True if `arg` acts as a list and does not look like a string."""
    string_behaviour = (
        isinstance(arg, six.string_types) or
        isinstance(arg, six.text_type))
    list_behaviour = hasattr(arg, '__getitem__') or hasattr(arg, '__iter__')
    return not string_behaviour and list_behaviour