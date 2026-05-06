def on_uppercase(self):
        """ Uppercase key press handler. """
        self.uppercase = not self.uppercase
        self.original_layout.set_uppercase(self.uppercase)
        self.special_char_layout.set_uppercase(self.uppercase)
        self.invalidate()