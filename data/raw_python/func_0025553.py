def get_params(self):
        "Parameters used to initialize the class"
        import inspect
        a = inspect.getargspec(self.__init__)[0]
        out = dict()
        for key in a[1:]:
            value = getattr(self, "_%s" % key, None)
            out[key] = value
        return out