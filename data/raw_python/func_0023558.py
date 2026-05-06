def do_whatis(self, arg):
        """whatis arg
        Print the type of the argument.
        """
        try:
            value = self._getval(arg)
        except Exception:
            # _getval() already printed the error
            return
        code = None
        # Is it a function?
        try:
            code = value.__code__
        except Exception:
            pass
        if code:
            self.message('Function %s' % code.co_name)
            return
        # Is it an instance method?
        try:
            code = value.__func__.__code__
        except Exception:
            pass
        if code:
            self.message('Method %s' % code.co_name)
            return
        # Is it a class?
        if value.__class__ is type:
            self.message('Class %s.%s' % (value.__module__, value.__name__))
            return
        # None of the above...
        self.message(type(value))