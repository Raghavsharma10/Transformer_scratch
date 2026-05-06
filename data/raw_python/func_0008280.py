def set_bind(self):
        """
        Sets key bindings.
        """
        # Arrow keys and enter
        self.bind('<Up>', lambda e: self.on_key_press_repeat('Up'))
        self.bind('<Down>', lambda e: self.on_key_press_repeat('Down'))
        self.bind('<Shift-Up>', lambda e: self.on_key_press_repeat('Shift-Up'))
        self.bind('<Shift-Down>', lambda e: self.on_key_press_repeat('Shift-Down'))
        self.bind('<Control-Up>', lambda e: self.on_key_press_repeat('Control-Up'))
        self.bind('<Control-Down>', lambda e: self.on_key_press_repeat('Control-Down'))
        self.bind('<KeyRelease>', lambda e: self.on_key_release_repeat())

        # Mouse buttons: bit complex since they don't automatically
        # run in continuous mode like the arrow keys
        self.bind('<ButtonPress-1>', self._leftMouseDown)
        self.bind('<ButtonRelease-1>', self._leftMouseUp)
        self.bind('<Shift-ButtonPress-1>', self._shiftLeftMouseDown)
        self.bind('<Shift-ButtonRelease-1>', self._shiftLeftMouseUp)
        self.bind('<Control-Button-1>', lambda e: self.add(100))

        self.bind('<ButtonPress-3>', self._rightMouseDown)
        self.bind('<ButtonRelease-3>', self._rightMouseUp)
        self.bind('<Shift-ButtonPress-3>', self._shiftRightMouseDown)
        self.bind('<Shift-ButtonRelease-3>', self._shiftRightMouseUp)
        self.bind('<Control-Button-3>', lambda e: self.sub(100))

        self.bind('<Double-Button-1>', self._dadd1)
        self.bind('<Double-Button-3>', self._dsub1)
        self.bind('<Shift-Double-Button-1>', self._dadd10)
        self.bind('<Shift-Double-Button-3>', self._dsub10)
        self.bind('<Control-Double-Button-1>', self._dadd100)
        self.bind('<Control-Double-Button-3>', self._dsub100)

        self.bind('<Enter>', self._enter)