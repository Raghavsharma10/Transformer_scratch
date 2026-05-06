def do_retval(self, arg):
        """retval
        Print the return value for the last return of a function.
        """
        locals = self.get_locals(self.curframe)
        if '__return__' in locals:
            self.message(bdb.safe_repr(locals['__return__']))
        else:
            self.error('Not yet returned!')