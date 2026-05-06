def set_bind(self):
        """
        Sets key bindings -- we need this more than once
        """
        FloatEntry.set_bind(self)
        self.bind('<Next>', lambda e: self.set(self.fmin))
        self.bind('<Prior>', lambda e: self.set(self.fmax))