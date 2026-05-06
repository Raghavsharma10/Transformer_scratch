def do_display(self, arg):
        """display [expression]

        Display the value of the expression if it changed, each time execution
        stops in the current frame.

        Without expression, list all display expressions for the current frame.
        """
        if not arg:
            self.message('Currently displaying:')
            for item in self.displaying.get(self.curframe, {}).items():
                self.message('%s: %s' % bdb.safe_repr(item))
        else:
            val = self._getval_except(arg)
            self.displaying.setdefault(self.curframe, {})[arg] = val
            self.message('display %s: %s' % (arg, bdb.safe_repr(val)))