def set_bind(self):
        """
        Sets key bindings -- we need this more than once
        """
        RangedInt.set_bind(self)
        self.unbind('<Next>')
        self.unbind('<Prior>')
        self.bind('<Next>', lambda e: self.set(self._min()))
        self.bind('<Prior>', lambda e: self.set(self._max()))