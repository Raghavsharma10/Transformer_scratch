def set_unbind(self):
        """
        Unsets key bindings -- we need this more than once
        """
        IntegerEntry.set_unbind(self)
        self.unbind('<Button-1>')
        self.unbind('<Button-3>')
        self.unbind('<Up>')
        self.unbind('<Down>')
        self.unbind('<Enter>')
        self.unbind('<Next>')
        self.unbind('<Prior>')