def call_fdel(self, obj) -> None:
        """Remove the predefined custom value and call the delete function."""
        self.fdel(obj)
        try:
            del vars(obj)[self.name]
        except KeyError:
            pass