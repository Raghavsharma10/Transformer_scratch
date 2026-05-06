def set_bind(self):
        """
        Sets key bindings -- we need this more than once
        """
        IntegerEntry.set_bind(self)
        self.unbind('<Shift-Up>')
        self.unbind('<Shift-Down>')
        self.unbind('<Control-Up>')
        self.unbind('<Control-Down>')
        self.unbind('<Double-Button-1>')
        self.unbind('<Double-Button-3>')
        self.unbind('<Shift-Button-1>')
        self.unbind('<Shift-Button-3>')
        self.unbind('<Control-Button-1>')
        self.unbind('<Control-Button-3>')

        self.bind('<Button-1>', lambda e: self.add(1))
        self.bind('<Button-3>', lambda e: self.sub(1))
        self.bind('<Up>', lambda e: self.add(1))
        self.bind('<Down>', lambda e: self.sub(1))
        self.bind('<Enter>', self._enter)
        self.bind('<Next>', lambda e: self.set(self.allowed[0]))
        self.bind('<Prior>', lambda e: self.set(self.allowed[-1]))