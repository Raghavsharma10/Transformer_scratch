def getter_(self, fget) -> 'BaseProperty':
        """Add the given getter function and its docstring to the
         property and return it."""
        self.fget = fget
        self.set_doc(fget.__doc__)
        return self