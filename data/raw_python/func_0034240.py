def on_special_char(self):
        """ Special char key press handler. """
        self.special_char = not self.special_char
        if self.special_char:
            self.set_layout(self.special_char_layout)
        else:
            self.set_layout(self.original_layout)
        self.invalidate()