def strippen(function):
    "Decorator. Strip excess whitespace from return value."
    def wrapper(*args, **kwargs):
        return strip_strings(function(*args, **kwargs))
    return wrapper