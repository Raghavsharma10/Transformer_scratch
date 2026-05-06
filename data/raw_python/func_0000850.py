def nullify(function):
    "Decorator. If empty list, returns None, else list."
    def wrapper(*args, **kwargs):
        value = function(*args, **kwargs)
        if(type(value) == list and len(value) == 0):
            return None
        return value
    return wrapper