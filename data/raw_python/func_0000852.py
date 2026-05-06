def inten(function):
    "Decorator. Attempts to convert return value to int"
    def wrapper(*args, **kwargs):
        return coerce_to_int(function(*args, **kwargs))
    return wrapper