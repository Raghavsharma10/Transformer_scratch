def set_bind(self):
        """
        Sets key bindings -- we need this more than once
        """
        IntegerEntry.set_bind(self)
        self.bind('<Next>', lambda e: self.set(self.imin))
        self.bind('<Prior>', lambda e: self.set(self.imax))