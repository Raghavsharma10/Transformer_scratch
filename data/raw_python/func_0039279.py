def fsto(self):
        """sto operation.
        """
        a = float(self.tmpopslist.pop())
        var = self.opslist.pop()
        if isinstance(var, basestring):
            self.variables.update({var: a})
            return a
        else:
            print("Can only sto into a variable.")
            return 'ERROR'