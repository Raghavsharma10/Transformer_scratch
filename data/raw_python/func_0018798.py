def call_fset(self, obj, value) -> None:
        """Store the given custom value and call the setter function."""
        vars(obj)[self.name] = self.fset(obj, value)