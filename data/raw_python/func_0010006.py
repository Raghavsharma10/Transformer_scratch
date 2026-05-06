def requires_private_key(func):
    """
    Decorator for functions that require the private key to be defined.
    """

    def func_wrapper(self, *args, **kwargs):
        if hasattr(self, "_DiffieHellman__private_key"):
            func(self, *args, **kwargs)
        else:
            self.generate_private_key()
            func(self, *args, **kwargs)

    return func_wrapper