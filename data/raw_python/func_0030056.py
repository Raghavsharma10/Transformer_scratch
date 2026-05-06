def required(field):
    """Decorator that checks if return value is set, if not, raises exception.
    """

    def wrap(f):
        def wrappedf(*args):
            result = f(*args)
            if result is None or result == "":
                raise Exception(
                    "Config option '%s' is required." % field)
            else:
                return result
        return wrappedf
    return wrap