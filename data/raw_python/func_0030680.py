def call_interval(freq, **kwargs):
    """Decorator for the CallInterval wrapper"""
    def wrapper(f):
        return CallInterval(f, freq, **kwargs)

    return wrapper