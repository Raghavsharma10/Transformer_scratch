def one_of(*args):
    """
    Validates that a field value matches one of the values
    given to this validator.
    """
    if len(args) == 1 and isinstance(args[0], list):
        items = args[0]
    else:
        items = list(args)

    def validate(value):
        if not value in items:
            return e("{} is not in the list {}", value, items)
    return validate