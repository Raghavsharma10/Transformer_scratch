def _arg_repr(self, arg):
        """ Get a useful (and not too large) represetation of an argument.
        """
        r = repr(arg)
        max = 40
        if len(r) > max:
            if hasattr(arg, 'shape'):
                r = 'array:' + 'x'.join([repr(s) for s in arg.shape])
            else:
                r = r[:max-3] + '...'
        return r