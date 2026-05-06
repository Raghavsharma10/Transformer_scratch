def do_args(self, arg):
        """a(rgs)
        Print the argument list of the current function.
        """
        co = self.curframe.f_code
        dict = self.get_locals(self.curframe)
        n = co.co_argcount
        if co.co_flags & 4: n = n+1
        if co.co_flags & 8: n = n+1
        for i in range(n):
            name = co.co_varnames[i]
            if name in dict:
                self.message('%s = %s' % (name, bdb.safe_repr(dict[name])))
            else:
                self.message('%s = *** undefined ***' % (name,))