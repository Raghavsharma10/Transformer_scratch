def extraBindingsForSelectableText(self):
        """ Collect in 1 place the bindings needed for watchTextSelection() """
        # See notes in watchTextSelection
        self.entry.bind('<FocusIn>', self.watchTextSelection, "+")
        self.entry.bind('<ButtonRelease-1>', self.watchTextSelection, "+")
        self.entry.bind('<B1-Motion>', self.watchTextSelection, "+")
        self.entry.bind('<Shift_L>', self.watchTextSelection, "+")
        self.entry.bind('<Left>', self.watchTextSelection, "+")
        self.entry.bind('<Right>', self.watchTextSelection, "+")