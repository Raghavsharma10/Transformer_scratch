def submit_btn(self, value, success=None):
        """This presses an input button with type=submit.
        Success must be given as a tuple of a (coordinate, timeout).
        Use (coordinate,) if you want to use the default timeout."""
        self.press("css=input[value='{}']".format(value))
        if success is not None:
            assert self.is_available(*success)