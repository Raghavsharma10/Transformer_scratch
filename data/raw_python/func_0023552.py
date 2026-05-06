def do_p(self, arg):
        """p expression
        Print the value of the expression.
        """
        try:
            self.message(bdb.safe_repr(self._getval(arg)))
        except Exception:
            pass