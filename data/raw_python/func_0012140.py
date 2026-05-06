def join(C, *args, **kwargs):
        """join a list of url elements, and include any keyword arguments, as a new URL"""
        u = C('/'.join([str(arg).strip('/') for arg in args]), **kwargs)
        return u