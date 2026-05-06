def updater(f):
    "Decorate a function with named arguments into updater for transact"
    @functools.wraps(f)
    def wrapped_updater(keys, values):
        result = f(*values)
        return (keys[:len(result)], result)
    return wrapped_updater