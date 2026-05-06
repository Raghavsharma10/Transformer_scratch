def local_property():
    """ Property structure which maps within the :func:local() thread
        (c)2014, Marcel Hellkamp
    """
    ls = local()

    def fget(self):
        try:
            return ls.var
        except AttributeError:
            raise RuntimeError("Request context not initialized.")

    def fset(self, value):
        ls.var = value

    def fdel(self):
        del ls.var

    return property(fget, fset, fdel, 'Thread-local property')