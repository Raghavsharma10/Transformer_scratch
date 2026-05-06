def set_bind(self):
        """
        Sets key bindings.
        """
        self.bind('<Button-1>', lambda e: self.add(0.1))
        self.bind('<Button-3>', lambda e: self.sub(0.1))
        self.bind('<Up>', lambda e: self.add(0.1))
        self.bind('<Down>', lambda e: self.sub(0.1))
        self.bind('<Shift-Up>', lambda e: self.add(1))
        self.bind('<Shift-Down>', lambda e: self.sub(1))
        self.bind('<Control-Up>', lambda e: self.add(10))
        self.bind('<Control-Down>', lambda e: self.sub(10))
        self.bind('<Double-Button-1>', self._dadd)
        self.bind('<Double-Button-3>', self._dsub)
        self.bind('<Shift-Button-1>', lambda e: self.add(1))
        self.bind('<Shift-Button-3>', lambda e: self.sub(1))
        self.bind('<Control-Button-1>', lambda e: self.add(10))
        self.bind('<Control-Button-3>', lambda e: self.sub(10))
        self.bind('<Enter>', self._enter)