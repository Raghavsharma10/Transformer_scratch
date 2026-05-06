def press(self, coordinate, success=None):
        """Success must be given as a tuple of a (coordinate, timeout).
        Use (coordinate,) if you want to use the default timeout."""
        if isinstance(coordinate, WebElement):
            coordinate.click()
        else:
            self.get_element(coordinate).click()
        if success is not None:
            assert self.is_available(*success)