def do_pp(self, arg):
        """pp expression
        Pretty-print the value of the expression.
        """
        obj = self._getval(arg)
        try:
            repr(obj)
        except Exception:
            self.message(bdb.safe_repr(obj))
        else:
            self.message(pprint.pformat(obj))